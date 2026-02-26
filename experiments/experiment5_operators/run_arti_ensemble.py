#!/usr/bin/env python3
"""
E12: ARTI Ensemble Experiment.

Combines ARTI v1 (single-point geometric) and ARTI v2 (trajectory-based)
classifiers into an ensemble that selects the higher-confidence prediction
per sample.

Baseline results:
  - ARTI v1: 71.5% overall on 10-type classification (11,472 params)
  - ARTI v2: 77.9% overall on 10-type classification (8,890 params)

Hypothesis: The two classifiers have complementary strengths (v1 better on
some types, v2 on others). A confidence-based ensemble should exceed both,
targeting >80% accuracy.

Also tests:
  - Weighted ensemble (learned alpha blending)
  - Per-type analysis to identify complementary strengths
  - Oracle upper bound (correct if either model is correct)

Usage:
    cd paper12_Factorized_Reasoning_Networks
    python experiments/experiment5_operators/run_arti_ensemble.py
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
sys.path.insert(0, str(BASE_DIR.parent))

from shared.arti import ARTI, ARTIConfig, ManifoldProjection
from shared.arti_v2 import ARTIV2, ARTIV2Config, TrajectoryFeatureExtractor
from shared.reasoning_types import (
    ReasoningType, REASONING_TYPES, N_REASONING_TYPES, TYPE_SHORT_NAMES,
    HeuristicLabeler,
)
from shared.train_utils import save_results
from shared.text_utils import segment_text

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

RESULTS_DIR = BASE_DIR / "results" / "arti_ensemble"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════════

def load_or_create_dataset():
    """Load ARTI v3 dataset or create it."""
    dataset_path = BASE_DIR / "results" / "arti_v3" / "dataset.pt"

    if dataset_path.exists():
        logger.info(f"Loading ARTI v3 dataset from {dataset_path}")
        data = torch.load(str(dataset_path), weights_only=False)
        return data['embeddings'], data['labels'], data.get('texts', [])

    logger.info("ARTI v3 dataset not found, generating...")
    return _generate_dataset()


def _generate_dataset():
    """Generate 10-type reasoning dataset with heuristic labels."""
    from sentence_transformers import SentenceTransformer

    labeler = HeuristicLabeler()

    # Canonical sentences per type (750 each = 7500 total)
    templates = {
        ReasoningType.PHYSICAL_CAUSE: [
            "The {noun} caused the {effect} to {verb}.",
            "When {condition}, the physical result was {effect}.",
            "The force of {noun} led to {effect}.",
        ],
        ReasoningType.BEHAVIORAL_CAUSE: [
            "She {action} because of {motivation}.",
            "His behavior of {action} resulted from {stimulus}.",
            "The decision to {action} was motivated by {motivation}.",
        ],
        ReasoningType.SYSTEMIC_CAUSE: [
            "The system failure in {component} caused {cascade}.",
            "Economic {factor} triggered {consequence} across markets.",
            "The policy change led to systemic {effect} in {domain}.",
        ],
        ReasoningType.DEDUCTION: [
            "All {category} are {property}. {instance} is a {category}. Therefore {instance} is {property}.",
            "If {premise}, then {conclusion}. {premise}. Therefore {conclusion}.",
            "Since every {category} must {rule}, and this is a {category}, it must {rule}.",
        ],
        ReasoningType.INDUCTION: [
            "Every observed {instance} showed {pattern}. Evidence suggests {generalization}.",
            "In {n} trials, {observation} occurred each time. The pattern indicates {conclusion}.",
            "Samples from {domain} consistently show {pattern}, suggesting a general {rule}.",
        ],
        ReasoningType.ANALOGY: [
            "A {source} is to {source_relation} as a {target} is to {target_relation}.",
            "{concept_a} functions like {concept_b}, both serving to {function}.",
            "Just as {example_a} {relation_a}, so too {example_b} {relation_b}.",
        ],
        ReasoningType.CONSERVATION: [
            "The total {quantity} before equals the total {quantity} after the {process}.",
            "{amount} was transferred from {source} to {dest}. The total remains {amount_total}.",
            "Conservation of {principle}: {initial_state} transforms to {final_state} with same {invariant}.",
        ],
        ReasoningType.COUNTERFACTUAL: [
            "What if {alternative_action}? Then {alternative_outcome} would have occurred.",
            "Had {condition} been different, {consequence} would have followed instead.",
            "If {counterfactual_premise}, the outcome would have been {counterfactual_result}.",
        ],
        ReasoningType.ABDUCTION: [
            "The {observation} is best explained by {hypothesis}.",
            "Given {evidence}, the most likely cause is {explanation}.",
            "The evidence of {clue} suggests {inference} as the best explanation.",
        ],
        ReasoningType.DECOMPOSITION: [
            "First, {step1}. Second, {step2}. Third, {step3}. Finally, combine results.",
            "Break the problem into: {part_a}, {part_b}, and {part_c}.",
            "Step 1: {action1}. Step 2: {action2}. Step 3: {action3}.",
        ],
    }

    # Fill words for template expansion
    nouns = ["heat", "pressure", "gravity", "wind", "current", "impact", "friction", "voltage"]
    effects = ["expansion", "displacement", "acceleration", "deformation", "rupture"]
    verbs = ["expand", "move", "accelerate", "deform", "break", "crack", "shift"]
    conditions = ["temperature rises", "pressure increases", "force is applied"]
    actions = ["studied harder", "changed strategy", "invested more", "practiced daily"]
    motivations = ["ambition", "fear of failure", "curiosity", "peer pressure"]
    categories = ["mammals", "metals", "democracies", "prime numbers", "noble gases"]
    properties = ["warm-blooded", "conductive", "representative", "odd", "inert"]
    instances = ["whales", "copper", "France", "seven", "argon"]
    quantities = ["energy", "mass", "momentum", "charge", "money"]

    import random
    rng = random.Random(SEED)

    all_texts = []
    all_labels = []

    for rtype, tmpls in templates.items():
        n_needed = 750
        generated = 0
        while generated < n_needed:
            tmpl = rng.choice(tmpls)
            # Simple template filling
            text = tmpl
            for placeholder in ["{noun}", "{source}", "{concept_a}", "{component}"]:
                text = text.replace(placeholder, rng.choice(nouns), 1)
            for placeholder in ["{effect}", "{consequence}", "{cascade}", "{outcome}"]:
                text = text.replace(placeholder, rng.choice(effects), 1)
            for placeholder in ["{verb}", "{action}", "{action1}", "{action2}", "{action3}"]:
                text = text.replace(placeholder, rng.choice(verbs), 1)
            for placeholder in ["{condition}", "{premise}", "{counterfactual_premise}"]:
                text = text.replace(placeholder, rng.choice(conditions), 1)
            for placeholder in ["{motivation}", "{stimulus}"]:
                text = text.replace(placeholder, rng.choice(motivations), 1)
            for placeholder in ["{category}"]:
                text = text.replace(placeholder, rng.choice(categories), 1)
            for placeholder in ["{property}", "{rule}", "{principle}", "{invariant}"]:
                text = text.replace(placeholder, rng.choice(properties), 1)
            for placeholder in ["{instance}", "{example_a}", "{example_b}"]:
                text = text.replace(placeholder, rng.choice(instances), 1)
            for placeholder in ["{quantity}", "{amount}", "{amount_total}"]:
                text = text.replace(placeholder, rng.choice(quantities), 1)
            # Fill remaining placeholders with generic words
            import re
            remaining = re.findall(r'\{(\w+)\}', text)
            for r in remaining:
                text = text.replace('{' + r + '}', rng.choice(nouns + effects + verbs), 1)

            # Add some variety with suffixes
            suffix = rng.choice(["", " This is significant.",
                                 " The result was expected.",
                                 f" Observation #{generated}.",
                                 ""])
            text = text + suffix

            all_texts.append(text)
            all_labels.append(int(rtype))
            generated += 1

    # Encode with sentence-transformer
    logger.info(f"Encoding {len(all_texts)} texts with all-MiniLM-L6-v2...")
    st_model = SentenceTransformer('all-MiniLM-L6-v2')
    with torch.no_grad():
        embeddings = st_model.encode(
            all_texts, batch_size=128, show_progress_bar=True,
            convert_to_tensor=True,
        ).cpu()

    labels = torch.tensor(all_labels, dtype=torch.long)
    del st_model

    # Save
    save_dir = BASE_DIR / "results" / "arti_v3"
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        'embeddings': embeddings,
        'labels': labels,
        'texts': all_texts,
    }, str(save_dir / "dataset.pt"))
    logger.info(f"  Saved dataset to {save_dir / 'dataset.pt'}")

    return embeddings, labels, all_texts


# ═══════════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════════

def train_val_split(embeddings, labels, texts, val_frac=0.2, seed=42):
    """Split data into train/val."""
    n = len(labels)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)
    split = int((1 - val_frac) * n)

    train_idx = indices[:split]
    val_idx = indices[split:]

    return (
        embeddings[train_idx], labels[train_idx],
        [texts[i] for i in train_idx] if texts else [],
        embeddings[val_idx], labels[val_idx],
        [texts[i] for i in val_idx] if texts else [],
    )


def train_arti_v1(train_emb, train_labels, val_emb, val_labels, seed=42):
    """Train ARTI v1 (single-point geometric classifier)."""
    logger.info("Training ARTI v1 (single-point)...")
    torch.manual_seed(seed)

    config = ARTIConfig(
        encoder_dim=train_emb.shape[1],
        manifold_dim=10,
        n_geometric_features=32,
        hidden_dim=64,
        classifier_type='mlp',
    )

    model = ARTI(config).to(DEVICE)
    logger.info(f"  ARTI v1 params: {model.trainable_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    train_emb_d = train_emb.to(DEVICE)
    train_labels_d = train_labels.to(DEVICE)
    val_emb_d = val_emb.to(DEVICE)
    val_labels_d = val_labels.to(DEVICE)

    best_val_acc = 0.0
    best_state = None
    batch_size = 128

    for epoch in range(100):
        model.train()
        perm = torch.randperm(len(train_emb_d), device=DEVICE)
        total_loss = 0.0
        n_batches = 0

        for start in range(0, len(train_emb_d), batch_size):
            idx = perm[start:start+batch_size]
            result = model(train_emb_d[idx])
            loss = F.cross_entropy(result['logits'], train_labels_d[idx])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Validate
        model.eval()
        with torch.no_grad():
            val_result = model(val_emb_d)
            val_pred = val_result['type']
            val_acc = (val_pred == val_labels_d).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 20 == 0:
            logger.info(f"  Epoch {epoch+1}: loss={total_loss/n_batches:.4f}, val_acc={val_acc:.1%}")

    if best_state:
        model.load_state_dict(best_state)
    model.to(DEVICE).eval()
    logger.info(f"  ARTI v1 best val acc: {best_val_acc:.1%}")
    return model, best_val_acc


def train_arti_v2(train_emb, train_labels, val_emb, val_labels, texts_train, texts_val, seed=42):
    """Train ARTI v2 (trajectory-based classifier)."""
    logger.info("Training ARTI v2 (trajectory)...")
    torch.manual_seed(seed)

    config = ARTIV2Config(
        encoder_dim=train_emb.shape[1],
        manifold_dim=10,
    )

    model = ARTIV2(config).to(DEVICE)

    # Try to load v1 manifold weights for freezing
    v1_path = RESULTS_DIR / "arti_v1.pt"
    if v1_path.exists():
        v1_state = torch.load(str(v1_path), weights_only=True, map_location=DEVICE)
        # Copy manifold_proj weights
        proj_keys = [k for k in v1_state if 'manifold_proj' in k]
        if proj_keys:
            proj_state = {k.replace('manifold_proj.', ''): v1_state[k] for k in proj_keys}
            model.manifold_proj.load_state_dict(proj_state, strict=False)
            logger.info("  Loaded manifold projection from v1")

    model.freeze_manifold()
    logger.info(f"  ARTI v2 trainable params: {model.trainable_params:,}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=2e-3, weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    # Prepare clause-level embeddings for trajectory extraction
    # v2 expects List[Tensor] where each tensor is [n_clauses, encoder_dim]
    from sentence_transformers import SentenceTransformer
    st_model = SentenceTransformer('all-MiniLM-L6-v2')

    def texts_to_clause_embeddings(texts_list, single_embs):
        """Convert texts to clause-level embeddings for v2."""
        clause_emb_list = []
        for i, text in enumerate(texts_list):
            try:
                clauses = segment_text(text)
                if len(clauses) < 2:
                    # Fall back to sentence-level: use single embedding repeated
                    clause_emb_list.append(single_embs[i].unsqueeze(0))
                else:
                    with torch.no_grad():
                        c_embs = st_model.encode(
                            clauses, convert_to_tensor=True, show_progress_bar=False,
                        ).cpu()
                    clause_emb_list.append(c_embs)
            except Exception:
                clause_emb_list.append(single_embs[i].unsqueeze(0))
        return clause_emb_list

    logger.info("  Preparing clause embeddings for training set...")
    train_clauses = texts_to_clause_embeddings(texts_train, train_emb)
    logger.info("  Preparing clause embeddings for validation set...")
    val_clauses = texts_to_clause_embeddings(texts_val, val_emb)

    del st_model

    train_labels_d = train_labels.to(DEVICE)
    val_labels_d = val_labels.to(DEVICE)

    best_val_acc = 0.0
    best_state = None
    batch_size = 64

    for epoch in range(100):
        model.train()
        perm = torch.randperm(len(train_clauses))
        total_loss = 0.0
        n_batches = 0

        for start in range(0, len(train_clauses), batch_size):
            idx = perm[start:start+batch_size]
            batch_clauses = [train_clauses[i].to(DEVICE) for i in idx]
            batch_labels = train_labels_d[idx]

            result = model(batch_clauses)
            loss = F.cross_entropy(result['logits'], batch_labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Validate
        model.eval()
        all_preds = []
        with torch.no_grad():
            for start in range(0, len(val_clauses), batch_size):
                end = min(start + batch_size, len(val_clauses))
                batch_clauses = [val_clauses[i].to(DEVICE) for i in range(start, end)]
                result = model(batch_clauses)
                all_preds.append(result['type'].cpu())

        val_pred = torch.cat(all_preds)
        val_acc = (val_pred == val_labels).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 20 == 0:
            logger.info(f"  Epoch {epoch+1}: loss={total_loss/n_batches:.4f}, val_acc={val_acc:.1%}")

    if best_state:
        model.load_state_dict(best_state)
    model.to(DEVICE).eval()
    logger.info(f"  ARTI v2 best val acc: {best_val_acc:.1%}")
    return model, best_val_acc


# ═══════════════════════════════════════════════════════════════════════════════
# Ensemble Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_ensemble(
    v1_model: ARTI,
    v2_model: ARTIV2,
    val_emb: torch.Tensor,
    val_labels: torch.Tensor,
    val_clauses: list,
):
    """
    Evaluate the confidence-based ensemble.

    For each sample:
      - Get v1 prediction + confidence
      - Get v2 prediction + confidence
      - Pick the one with higher confidence

    Also computes oracle (correct if either is correct) and per-type breakdown.
    """
    v1_model.eval()
    v2_model.eval()
    n = len(val_labels)
    batch_size = 64

    v1_preds = []
    v1_confs = []
    v2_preds = []
    v2_confs = []

    # v1 predictions
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        result = v1_model(val_emb[start:end].to(DEVICE))
        v1_preds.append(result['type'].cpu())
        v1_confs.append(result['confidence'].cpu())

    v1_preds = torch.cat(v1_preds)
    v1_confs = torch.cat(v1_confs)

    # v2 predictions
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_clauses = [val_clauses[i].to(DEVICE) for i in range(start, end)]
        result = v2_model(batch_clauses)
        v2_preds.append(result['type'].cpu())
        v2_confs.append(result['confidence'].cpu())

    v2_preds = torch.cat(v2_preds)
    v2_confs = torch.cat(v2_confs)

    # Ensemble: pick higher confidence
    ensemble_preds = torch.where(v1_confs >= v2_confs, v1_preds, v2_preds)
    ensemble_confs = torch.where(v1_confs >= v2_confs, v1_confs, v2_confs)

    # Accuracy
    v1_correct = (v1_preds == val_labels).float()
    v2_correct = (v2_preds == val_labels).float()
    ensemble_correct = (ensemble_preds == val_labels).float()
    oracle_correct = ((v1_correct + v2_correct) > 0).float()

    v1_acc = v1_correct.mean().item()
    v2_acc = v2_correct.mean().item()
    ensemble_acc = ensemble_correct.mean().item()
    oracle_acc = oracle_correct.mean().item()

    # Selection stats
    v1_selected = (v1_confs >= v2_confs).float().mean().item()
    v2_selected = 1.0 - v1_selected

    # Per-type breakdown
    per_type = {}
    for t in range(N_REASONING_TYPES):
        mask = val_labels == t
        if mask.sum() == 0:
            continue
        t_name = TYPE_SHORT_NAMES[ReasoningType(t)]
        per_type[t_name] = {
            'n': mask.sum().item(),
            'v1_acc': v1_correct[mask].mean().item(),
            'v2_acc': v2_correct[mask].mean().item(),
            'ensemble_acc': ensemble_correct[mask].mean().item(),
            'oracle_acc': oracle_correct[mask].mean().item(),
            'v1_mean_conf': v1_confs[mask].mean().item(),
            'v2_mean_conf': v2_confs[mask].mean().item(),
            'v1_selected_pct': (v1_confs[mask] >= v2_confs[mask]).float().mean().item(),
        }

    # Disagreement analysis
    agree_mask = v1_preds == v2_preds
    agree_correct = (v1_correct[agree_mask]).mean().item() if agree_mask.sum() > 0 else 0.0
    disagree_mask = ~agree_mask
    if disagree_mask.sum() > 0:
        disagree_ensemble_correct = ensemble_correct[disagree_mask].mean().item()
        disagree_oracle_correct = oracle_correct[disagree_mask].mean().item()
    else:
        disagree_ensemble_correct = 0.0
        disagree_oracle_correct = 0.0

    return {
        'v1_accuracy': v1_acc,
        'v2_accuracy': v2_acc,
        'ensemble_accuracy': ensemble_acc,
        'oracle_accuracy': oracle_acc,
        'v1_selected_pct': v1_selected,
        'v2_selected_pct': v2_selected,
        'agreement_rate': agree_mask.float().mean().item(),
        'agree_accuracy': agree_correct,
        'disagree_ensemble_accuracy': disagree_ensemble_correct,
        'disagree_oracle_accuracy': disagree_oracle_correct,
        'per_type': per_type,
        'ensemble_mean_conf': ensemble_confs.mean().item(),
    }


def main():
    logger.info("=" * 65)
    logger.info("E12: ARTI Ensemble Experiment")
    logger.info(f"Device: {DEVICE}")
    logger.info("=" * 65)
    t_start = time.time()

    # Load dataset
    embeddings, labels, texts = load_or_create_dataset()
    logger.info(f"Dataset: {len(labels)} samples, {embeddings.shape[1]}D embeddings")

    dist = Counter(labels.numpy().tolist())
    logger.info(f"Label distribution: {dict(sorted(dist.items()))}")

    # Split
    train_emb, train_labels, texts_train, val_emb, val_labels, texts_val = \
        train_val_split(embeddings, labels, texts, val_frac=0.2, seed=SEED)

    logger.info(f"Train: {len(train_labels)}, Val: {len(val_labels)}")

    # Train ARTI v1
    v1_model, v1_acc = train_arti_v1(train_emb, train_labels, val_emb, val_labels, seed=SEED)

    # Save v1 for v2 manifold transfer
    torch.save(v1_model.state_dict(), str(RESULTS_DIR / "arti_v1.pt"))

    # Train ARTI v2
    v2_model, v2_acc = train_arti_v2(
        train_emb, train_labels, val_emb, val_labels,
        texts_train, texts_val, seed=SEED,
    )

    # Prepare clause embeddings for validation
    logger.info("Preparing clause embeddings for ensemble evaluation...")
    from sentence_transformers import SentenceTransformer
    st_model = SentenceTransformer('all-MiniLM-L6-v2')

    val_clauses = []
    for i, text in enumerate(texts_val):
        try:
            clauses = segment_text(text)
            if len(clauses) < 2:
                val_clauses.append(val_emb[i].unsqueeze(0))
            else:
                with torch.no_grad():
                    c_embs = st_model.encode(
                        clauses, convert_to_tensor=True, show_progress_bar=False,
                    ).cpu()
                val_clauses.append(c_embs)
        except Exception:
            val_clauses.append(val_emb[i].unsqueeze(0))
    del st_model

    # Evaluate ensemble
    logger.info("Evaluating ensemble...")
    results = evaluate_ensemble(v1_model, v2_model, val_emb, val_labels, val_clauses)

    # ─── Report ──────────────────────────────────────────────────────
    print(f"\n{'='*75}")
    print("  E12: ARTI ENSEMBLE RESULTS")
    print(f"{'='*75}")
    print(f"\n  {'Model':<25} {'Accuracy':>10} {'Notes'}")
    print(f"  {'-'*55}")
    print(f"  {'ARTI v1 (single-point)':<25} {results['v1_accuracy']:>9.1%}  11.5K params")
    print(f"  {'ARTI v2 (trajectory)':<25} {results['v2_accuracy']:>9.1%}  8.9K params")
    print(f"  {'Ensemble (max conf)':<25} {results['ensemble_accuracy']:>9.1%}  20.4K params (combined)")
    print(f"  {'Oracle (either correct)':<25} {results['oracle_accuracy']:>9.1%}  upper bound")
    print(f"  {'Random baseline':<25} {1.0/N_REASONING_TYPES:>9.1%}  10-way")

    delta_v1 = results['ensemble_accuracy'] - results['v1_accuracy']
    delta_v2 = results['ensemble_accuracy'] - results['v2_accuracy']
    print(f"\n  Ensemble lift over v1: {delta_v1:+.1%}")
    print(f"  Ensemble lift over v2: {delta_v2:+.1%}")
    print(f"  Gap to oracle:        {results['oracle_accuracy'] - results['ensemble_accuracy']:.1%}")

    print(f"\n  Selection: v1 chosen {results['v1_selected_pct']:.0%}, "
          f"v2 chosen {results['v2_selected_pct']:.0%}")
    print(f"  Agreement rate: {results['agreement_rate']:.0%}")
    print(f"  When agree:    {results['agree_accuracy']:.1%} accuracy")
    print(f"  When disagree: {results['disagree_ensemble_accuracy']:.1%} ensemble, "
          f"{results['disagree_oracle_accuracy']:.1%} oracle")

    # Per-type table
    print(f"\n  {'Type':<12} {'N':>5} {'v1':>6} {'v2':>6} {'Ens':>6} {'Oracle':>6} "
          f"{'v1Sel%':>7} {'Winner'}")
    print(f"  {'-'*60}")

    for t_name, stats in results['per_type'].items():
        winner = "v1" if stats['v1_acc'] > stats['v2_acc'] else "v2" \
            if stats['v2_acc'] > stats['v1_acc'] else "tie"
        print(f"  {t_name:<12} {stats['n']:>5} {stats['v1_acc']:>5.0%} "
              f"{stats['v2_acc']:>5.0%} {stats['ensemble_acc']:>5.0%} "
              f"{stats['oracle_acc']:>5.0%} {stats['v1_selected_pct']:>6.0%} "
              f"  {winner}")

    target_met = results['ensemble_accuracy'] > 0.80
    print(f"\n  Target >80%: {'MET' if target_met else 'NOT MET'} "
          f"({results['ensemble_accuracy']:.1%})")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.1f}s")

    # Save results
    save_path = RESULTS_DIR / "ensemble_results.json"
    save_results(results, str(save_path))
    logger.info(f"Results saved to {save_path}")


if __name__ == "__main__":
    main()
