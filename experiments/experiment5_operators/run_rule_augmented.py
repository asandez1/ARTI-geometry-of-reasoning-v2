#!/usr/bin/env python3
"""
Option B: Rule-Augmented Benchmark Slice.

Tests whether injecting external rule knowledge via RuleInjectionLayer
can lift CO-FRN performance on non-math benchmarks (StrategyQA, FOLIO).

Architecture:
    q_emb [B, 384] -> COFRN.encode_text() -> s0 [B, 256]  (frozen)
                    -> RuleLibrary.retrieve(q_emb, top_k=3) -> rule_context [B, 384]
                    -> RuleInjectionLayer(s0, rule_context) -> s0_aug [B, 256]  (trainable, ~98K)
                    -> COFRN.factorization(s0_aug) -> ... -> scores  (frozen)

Training: GSM8K + ARC train, curriculum depth 1→2→3, only RuleInjectionLayer trained.
Evaluation: 100 StrategyQA + 100 FOLIO + 100 GSM8K + 100 ARC (seed=42).

Ablations:
    1. no_rules:        skip rule injection (baseline)
    2. random_rules:    random rule embeddings instead of retrieved
    3. domain_filtered: only retrieve rules from matching domain
    4. top_k sweep:     top_k ∈ {1, 3, 5}

Usage:
    cd paper12_Factorized_Reasoning_Networks
    python experiments/experiment5_operators/run_rule_augmented.py
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
from shared.rule_injection import RuleLibrary, RuleInjectionLayer, RuleAugmentedCOFRN
from shared.data_utils import (
    precompute_embeddings_st, BenchmarkDataset, MixedDomainDataset,
)
from shared.train_utils import TrainConfig, CurriculumPhase, save_results
from shared.metrics import anchor_utilization_entropy

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

RESULTS_DIR = BASE_DIR / "results" / "rule_augmented"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CONTROLLER_DIR = BASE_DIR / "results" / "controller"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42


# ═══════════════════════════════════════════════════════════════════════════════
# Load Pre-Trained COFRN
# ═══════════════════════════════════════════════════════════════════════════════

def load_cofrn() -> COFRN:
    """Load the Phase 0 pre-trained COFRN (from run_controller.py)."""
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
    if not model_path.exists():
        logger.error(f"COFRN model not found at {model_path}")
        logger.error("Run `python experiments/experiment5_operators/run_controller.py` first.")
        sys.exit(1)

    cofrn = COFRN(config)
    cofrn.load_state_dict(
        torch.load(str(model_path), weights_only=True, map_location=DEVICE)
    )
    cofrn.to(DEVICE).eval()
    logger.info(f"COFRN loaded: {cofrn.trainable_params:,} params (will be frozen)")
    return cofrn


# ═══════════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════════

def train_rule_augmented(
    cofrn: COFRN,
    rule_library: RuleLibrary,
    top_k: int = 3,
    use_rules: bool = True,
    use_random_rules: bool = False,
    domain_filter: str = None,
    tag: str = "default",
) -> RuleAugmentedCOFRN:
    """
    Train the RuleInjectionLayer on GSM8K + ARC.

    COFRN is frozen. Only ~98K params in the injection layer are trained.
    Uses curriculum: depth 1 (5 ep) → 2 (5 ep) → 3 (10 ep).
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Training rule-augmented COFRN [{tag}]")
    logger.info(f"  use_rules={use_rules}, random_rules={use_random_rules}, "
                f"top_k={top_k}, domain_filter={domain_filter}")
    logger.info(f"{'='*60}")

    # Build model
    model = RuleAugmentedCOFRN(
        cofrn=cofrn,
        rule_library=rule_library,
        rule_dim=rule_library.emb_dim,
        hidden_dim=cofrn.config.hidden_dim,
        alpha_init=0.1,
        top_k=top_k,
        use_rules=use_rules,
        use_random_rules=use_random_rules,
        domain_filter=domain_filter,
    ).to(DEVICE)

    logger.info(f"  Trainable params: {model.trainable_params:,}")

    # Load training data
    logger.info("  Loading training data (GSM8K + ARC)...")
    try:
        gsm8k_ds = precompute_embeddings_st(
            'all-MiniLM-L6-v2', 'gsm8k', 'train', seed=SEED,
        )
        arc_ds = precompute_embeddings_st(
            'all-MiniLM-L6-v2', 'arc_challenge', 'train', seed=SEED,
        )
        mixed_ds = MixedDomainDataset({'gsm8k': gsm8k_ds, 'arc_challenge': arc_ds})
        n = len(mixed_ds)
    except Exception as e:
        logger.error(f"  Could not load training data: {e}")
        logger.info("  Using random data fallback...")
        n = 1000
        torch.manual_seed(SEED)
        mixed_q = torch.randn(n, 384)
        mixed_a = torch.randn(n, 4, 384)
        mixed_l = torch.randint(0, 4, (n,))
        mixed_ds = type('FakeDS', (), {
            'question_embeddings': mixed_q,
            'answer_embeddings': mixed_a,
            'labels': mixed_l,
        })()

    # Train/val split
    val_size = min(500, n // 5)
    rng = np.random.RandomState(SEED)
    indices = rng.permutation(n)
    train_idx = indices[val_size:]
    val_idx = indices[:val_size]

    train_q = mixed_ds.question_embeddings[train_idx]
    train_a = mixed_ds.answer_embeddings[train_idx]
    train_l = mixed_ds.labels[train_idx]
    val_q = mixed_ds.question_embeddings[val_idx]
    val_a = mixed_ds.answer_embeddings[val_idx]
    val_l = mixed_ds.labels[val_idx]

    logger.info(f"  Train: {len(train_idx)}, Val: {len(val_idx)}")

    # Curriculum phases
    phases = [
        {'depth': 1, 'epochs': 5, 'lr': 3e-4},
        {'depth': 2, 'epochs': 5, 'lr': 1e-4},
        {'depth': 3, 'epochs': 10, 'lr': 5e-5},
    ]

    batch_size = 32
    best_val_acc = 0.0
    best_state = None
    alpha_trajectory = []
    total_epoch = 0

    for phase in phases:
        depth = phase['depth']
        lr = phase['lr']
        n_epochs = phase['epochs']

        logger.info(f"\n  Phase: depth={depth}, lr={lr}, epochs={n_epochs}")

        # Only optimize injection layer params
        optimizer = torch.optim.AdamW(
            model.injection.parameters(),
            lr=lr, weight_decay=0.01,
        )

        for epoch in range(n_epochs):
            model.train()
            perm = torch.randperm(len(train_idx))
            total_loss = 0.0
            total_correct = 0
            total_seen = 0

            for start in range(0, len(train_idx), batch_size):
                idx = perm[start:start + batch_size]
                q = train_q[idx].to(DEVICE)
                a = train_a[idx].to(DEVICE)
                l = train_l[idx].to(DEVICE)

                out = model(
                    embeddings=q,
                    answer_embeddings=a,
                    labels=l,
                    depth=depth,
                )

                loss = out['total_loss']
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.injection.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item() * len(idx)
                if 'correct' in out:
                    total_correct += out['correct'].sum().item()
                total_seen += len(idx)

            train_loss = total_loss / max(total_seen, 1)
            train_acc = total_correct / max(total_seen, 1)

            # Validate
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for start in range(0, len(val_idx), batch_size):
                    end = min(start + batch_size, len(val_idx))
                    q = val_q[start:end].to(DEVICE)
                    a = val_a[start:end].to(DEVICE)
                    l = val_l[start:end].to(DEVICE)

                    out = model(embeddings=q, answer_embeddings=a, labels=l, depth=depth)
                    if 'correct' in out:
                        val_correct += out['correct'].sum().item()
                    val_total += len(l)

            val_acc = val_correct / max(val_total, 1)
            alpha_val = model.injection.alpha.item()
            alpha_trajectory.append({
                'epoch': total_epoch,
                'depth': depth,
                'alpha': alpha_val,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
            })

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.injection.state_dict().items()}

            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(
                    f"    Ep {total_epoch:>2}: loss={train_loss:.4f} "
                    f"train_acc={train_acc:.1%} val_acc={val_acc:.1%} "
                    f"alpha={alpha_val:.4f}"
                )

            total_epoch += 1

    # Restore best
    if best_state is not None:
        model.injection.load_state_dict(best_state)
    model.to(DEVICE).eval()

    logger.info(f"  Best val acc: {best_val_acc:.1%}")
    logger.info(f"  Final alpha: {model.injection.alpha.item():.4f}")

    # Save injection layer weights and alpha trajectory
    save_path = RESULTS_DIR / f"injection_{tag}.pt"
    torch.save(model.injection.state_dict(), str(save_path))

    save_results(
        {'alpha_trajectory': alpha_trajectory, 'best_val_acc': best_val_acc, 'tag': tag},
        str(RESULTS_DIR / f"train_{tag}.json"),
    )

    return model


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_model_on_benchmark(
    model: nn.Module,
    benchmark: str,
    split: str,
    n_examples: int,
    cofrn_baseline: COFRN = None,
    depth: int = 3,
) -> dict:
    """
    Evaluate a model (rule-augmented or baseline) on a benchmark slice.

    Returns dict with accuracy and diagnostics.
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

    # Evaluate model
    model.eval()
    correct = 0
    all_anchor_weights = []
    all_rule_info = []

    with torch.no_grad():
        for start in range(0, n, 32):
            end = min(start + 32, n)
            out = model(
                embeddings=q_emb[start:end],
                answer_embeddings=a_emb[start:end],
                labels=labels[start:end],
                depth=depth,
            )
            if 'correct' in out:
                correct += out['correct'].sum().item()
            if 'anchor_weights' in out:
                all_anchor_weights.append(out['anchor_weights'].cpu())
            if 'rule_info' in out and out['rule_info'] is not None:
                all_rule_info.extend(out['rule_info'])

    acc = correct / n

    # Anchor entropy
    anchor_ent = None
    if all_anchor_weights:
        aw = torch.cat(all_anchor_weights, dim=0).numpy()
        ent_stats = anchor_utilization_entropy(aw, n_anchors=16)
        anchor_ent = ent_stats['entropy']

    # COFRN baseline accuracy (if provided)
    cofrn_acc = None
    if cofrn_baseline is not None:
        cofrn_correct = 0
        with torch.no_grad():
            for start in range(0, n, 32):
                end = min(start + 32, n)
                out_base = cofrn_baseline(
                    embeddings=q_emb[start:end],
                    answer_embeddings=a_emb[start:end],
                    labels=labels[start:end],
                )
                pred = out_base['scores'].argmax(dim=-1)
                cofrn_correct += (pred == labels[start:end]).sum().item()
        cofrn_acc = cofrn_correct / n

    result = {
        'benchmark': benchmark,
        'split': split,
        'n_examples': n,
        'n_choices': n_choices,
        'random_baseline': random_baseline,
        'accuracy': acc,
        'cofrn_baseline_acc': cofrn_acc,
        'anchor_entropy': anchor_ent,
    }

    # Sample rule retrievals (first 3 examples)
    if all_rule_info and len(all_rule_info) >= 3:
        result['sample_retrievals'] = all_rule_info[:3]

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    logger.info("=" * 65)
    logger.info("  Option B: Rule-Augmented Benchmark Slice")
    logger.info(f"  Device: {DEVICE}")
    logger.info("=" * 65)
    t0 = time.time()

    # Load pre-trained COFRN
    cofrn = load_cofrn()

    # Build rule library
    logger.info("\nBuilding RuleLibrary...")
    rule_library = RuleLibrary(device=DEVICE)

    # ─── Condition 1: COFRN Baseline (no rules) ──────────────────────
    logger.info("\n" + "=" * 65)
    logger.info("  Condition 1: COFRN Baseline (no rule injection)")
    logger.info("=" * 65)

    model_no_rules = RuleAugmentedCOFRN(
        cofrn=cofrn,
        rule_library=rule_library,
        use_rules=False,
    ).to(DEVICE)
    model_no_rules.eval()

    # ─── Condition 2: Rule-Augmented COFRN (main experiment) ─────────
    model_rules = train_rule_augmented(
        cofrn, rule_library,
        top_k=3, use_rules=True, use_random_rules=False,
        tag="rules_topk3",
    )

    # ─── Condition 3: Random Rules Ablation ──────────────────────────
    model_random = train_rule_augmented(
        cofrn, rule_library,
        top_k=3, use_rules=True, use_random_rules=True,
        tag="random_rules",
    )

    # ─── Evaluation on all 4 benchmarks ──────────────────────────────
    benchmarks = [
        ('gsm8k', 'test', 100),
        ('arc_challenge', 'test', 100),
        ('strategyqa', 'train', 100),
        ('folio', 'validation', 100),
    ]

    conditions = {
        'no_rules': model_no_rules,
        'rules_topk3': model_rules,
        'random_rules': model_random,
    }

    all_results = []

    for bm, split, n in benchmarks:
        logger.info(f"\n{'='*65}")
        logger.info(f"  Evaluating: {bm} ({split}, {n} examples)")
        logger.info(f"{'='*65}")

        for cond_name, cond_model in conditions.items():
            try:
                result = evaluate_model_on_benchmark(
                    cond_model, bm, split, n,
                    cofrn_baseline=cofrn if cond_name == 'no_rules' else None,
                )
                result['condition'] = cond_name
                all_results.append(result)

                logger.info(
                    f"  {cond_name:<16}: {result['accuracy']:.1%}"
                    f"  (anchor_ent={result.get('anchor_entropy', 'n/a')})"
                )
            except Exception as e:
                logger.error(f"  {cond_name}: FAILED ({e})")

    # ─── Top-k Ablation (on StrategyQA) ──────────────────────────────
    logger.info(f"\n{'='*65}")
    logger.info("  Top-k Ablation (StrategyQA)")
    logger.info(f"{'='*65}")

    topk_results = []
    for k in [1, 3, 5]:
        model_k = train_rule_augmented(
            cofrn, rule_library,
            top_k=k, use_rules=True, use_random_rules=False,
            tag=f"topk_{k}",
        )
        try:
            r = evaluate_model_on_benchmark(
                model_k, 'strategyqa', 'train', 100,
            )
            r['top_k'] = k
            topk_results.append(r)
            logger.info(f"  top_k={k}: {r['accuracy']:.1%}")
        except Exception as e:
            logger.error(f"  top_k={k}: FAILED ({e})")

    # ─── Summary Table ────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  OPTION B RESULTS: Rule-Augmented CO-FRN")
    print(f"{'='*80}")
    print(f"  {'Benchmark':<15} {'Random':>7} {'No Rules':>9} {'Rules(k=3)':>11} "
          f"{'RandRules':>10} {'Lift':>7}")
    print(f"  {'-'*70}")

    for bm, _, _ in benchmarks:
        bm_results = {r['condition']: r for r in all_results if r['benchmark'] == bm}
        no_rules = bm_results.get('no_rules', {})
        rules = bm_results.get('rules_topk3', {})
        random = bm_results.get('random_rules', {})

        rand_base = no_rules.get('random_baseline', 0)
        nr_acc = no_rules.get('accuracy', 0)
        r_acc = rules.get('accuracy', 0)
        rr_acc = random.get('accuracy', 0)
        lift = r_acc - nr_acc

        print(f"  {bm:<15} {rand_base:>6.1%} {nr_acc:>8.1%} {r_acc:>10.1%} "
              f"{rr_acc:>9.1%} {lift:>+6.1%}")

    if topk_results:
        print(f"\n  Top-k Ablation (StrategyQA):")
        for r in topk_results:
            print(f"    top_k={r['top_k']}: {r['accuracy']:.1%}")

    # ─── Diagnostics ──────────────────────────────────────────────────
    # Print sample rule retrievals for StrategyQA
    stqa_rules = [r for r in all_results
                  if r['benchmark'] == 'strategyqa' and r['condition'] == 'rules_topk3']
    if stqa_rules and 'sample_retrievals' in stqa_rules[0]:
        print(f"\n  Sample Rule Retrievals (StrategyQA):")
        for i, retrieval in enumerate(stqa_rules[0]['sample_retrievals'][:3]):
            print(f"    Example {i}:")
            for rule in retrieval:
                print(f"      {rule['rule_id']} {rule['name']:<30} "
                      f"({rule['domain']}) sim={rule['similarity']:.3f}")

    # Alpha trajectory for main model
    train_path = RESULTS_DIR / "train_rules_topk3.json"
    if train_path.exists():
        with open(str(train_path)) as f:
            train_data = json.load(f)
        traj = train_data.get('alpha_trajectory', [])
        if traj:
            print(f"\n  Alpha Trajectory (rules_topk3):")
            print(f"    Start: {traj[0]['alpha']:.4f}")
            print(f"    End:   {traj[-1]['alpha']:.4f}")
            print(f"    Best val acc: {train_data.get('best_val_acc', 0):.1%}")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")

    # Save all results
    save_results(
        {
            'benchmark_results': all_results,
            'topk_ablation': topk_results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'device': DEVICE,
        },
        str(RESULTS_DIR / "option_b_results.json"),
    )
    logger.info(f"Results saved to {RESULTS_DIR / 'option_b_results.json'}")


if __name__ == "__main__":
    main()
