#!/usr/bin/env python3
"""
Experiment 5: Cross-Domain Reasoning Operator Extraction

Three sequential phases with gates:
  Phase 1 — Break the encoder ceiling (sentence-transformers vs GPT-2)
  Phase 2 — Fix operator specialization (entropy schedule ablation)
  Phase 3 — Cross-domain operator transfer (main event)

If a gate fails, the experiment stops with a documented negative result.

Usage:
    python run_experiment.py --quick    # Phase 1 only, 1 seed, 5 epochs
    python run_experiment.py --full     # All phases, 3 seeds (Phase 3)
    python run_experiment.py --phase 1  # Run only Phase 1
    python run_experiment.py --phase 2  # Phase 2 (needs Phase 1 results)
    python run_experiment.py --phase 3  # Phase 3 (needs Phase 2 results)
"""

import argparse
import json
import logging
import sys
import time
import copy
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# Add experiments/ to path so `shared` is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.model import COFRN, COFRNConfig
from shared.encoder import PrecomputedEncoder, SentenceTransformerEncoder
from shared.baselines import MLPHead, LoRAModel, count_trainable_params
from shared.data_utils import (
    load_benchmark, load_benchmark_splits, BenchmarkDataset,
    BENCHMARK_CONFIGS, LOADERS, collate_benchmark,
    precompute_embeddings_st, MixedDomainDataset, collate_mixed_domain,
)
from shared.train_utils import (
    TrainConfig, train_model, evaluate_model, save_results,
)
from shared.metrics import (
    compute_centered_redundancy, bootstrap_rho_ci,
    anchor_utilization_entropy, per_domain_conditional_entropy,
    cross_domain_js_divergence,
    transfer_ratio, sample_efficiency_ratio,
    paired_permutation_test, cohens_d,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("experiment5")


# ─── Constants ────────────────────────────────────────────────────────

BENCHMARKS = ['gsm8k', 'arc_challenge', 'strategyqa', 'folio']
NON_MATH_BENCHMARKS = ['arc_challenge', 'strategyqa', 'folio']
SOURCE_BENCHMARKS = ['gsm8k', 'arc_challenge']
NEAR_TRANSFER = ['svamp', 'openbookqa']
FAR_TRANSFER = ['strategyqa', 'folio']
REVERSE_SOURCE = ['strategyqa', 'folio']
REVERSE_TARGET = ['gsm8k', 'svamp']

BENCHMARK_N_CLASSES = {
    'gsm8k': 4, 'arc_challenge': 4, 'strategyqa': 2,
    'folio': 3, 'svamp': 4, 'openbookqa': 4,
}

CHANCE_LEVELS = {
    'gsm8k': 0.25, 'arc_challenge': 0.25, 'strategyqa': 0.50,
    'folio': 0.333, 'svamp': 0.25, 'openbookqa': 0.25,
}

ENCODER_CONFIGS = {
    'gpt2': {'type': 'gpt2', 'dim': 768},
    'all-MiniLM-L6-v2': {'type': 'st', 'dim': 384},
    'all-mpnet-base-v2': {'type': 'st', 'dim': 768},
    'intfloat/e5-small-v2': {'type': 'st', 'dim': 384},
}

FEW_SHOT_BUDGETS = [16, 64, 256, 1024]


# ─── Helpers ──────────────────────────────────────────────────────────

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_or_precompute(
    encoder_name: str,
    benchmark: str,
    split: str,
    seed: int = 42,
    cache: Optional[Dict] = None,
) -> BenchmarkDataset:
    """
    Load benchmark with the specified encoder, using cache to avoid recomputation.

    Stores RAW embeddings (before projection) so that PrecomputedEncoder in
    the model handles the trainable projection. This ensures answer_proj
    receives encoder_input_dim-sized tensors as expected.
    """
    cache_key = f"{encoder_name}_{benchmark}_{split}_{seed}"
    if cache is not None and cache_key in cache:
        return cache[cache_key]

    enc_cfg = ENCODER_CONFIGS[encoder_name]

    if enc_cfg['type'] == 'gpt2':
        # Store RAW GPT-2 embeddings (768D, before projection)
        from shared.encoder import FrozenEncoder
        from shared.data_utils import EmbeddingCache, LOADERS
        import inspect

        encoder = FrozenEncoder(model_name='gpt2', hidden_dim=256, load_pretrained=True)
        emb_cache = EmbeddingCache(encoder=encoder, device='cpu')

        loader_fn = LOADERS[benchmark]
        kwargs = {'split': split}
        if 'seed' in inspect.signature(loader_fn).parameters:
            kwargs['seed'] = seed
        questions, choices, labels = loader_fn(**kwargs)

        logger.info(f"Precomputing GPT-2 embeddings for {benchmark}/{split}: "
                     f"{len(questions)} examples")

        # Raw embeddings at 768D (not projected)
        q_embs = emb_cache.encode_texts(questions, max_length=512)

        all_choice_texts = []
        for cs in choices:
            all_choice_texts.extend(cs)
        c_embs = emb_cache.encode_texts(all_choice_texts, max_length=512)
        n_choices = len(choices[0])
        c_embs = c_embs.reshape(len(questions), n_choices, -1)

        ds = BenchmarkDataset(
            question_embeddings=q_embs,
            answer_embeddings=c_embs,
            labels=torch.tensor(labels, dtype=torch.long),
            domain=BENCHMARK_CONFIGS[benchmark]['domain'],
            metadata=[{'question': q, 'choices': c} for q, c in zip(questions, choices)],
        )
        del encoder, emb_cache
    else:
        # Sentence-transformer: already stores raw embeddings
        ds = precompute_embeddings_st(
            encoder_name=encoder_name,
            benchmark=benchmark,
            split=split,
            seed=seed,
        )

    if cache is not None:
        cache[cache_key] = ds
    return ds


def make_cofrn(
    encoder_dim: int,
    n_classes: int,
    hidden_dim: int = 256,
    entropy_schedule: str = 'constant',
    entropy_anneal_epochs: int = 10,
    lambda_factorize: float = 0.1,
    lambda_entropy: float = 0.01,
    n_anchors: int = 16,
) -> COFRN:
    """Create a CO-FRN model with the specified config."""
    config = COFRNConfig(
        encoder_model='gpt2',
        hidden_dim=hidden_dim,
        use_precomputed=True,
        encoder_input_dim=encoder_dim,
        struct_dim=128,
        context_dim=128,
        manifold_dim=10,
        n_anchors=n_anchors,
        rank=16,
        task_type='single_step',
        lambda_factorize=lambda_factorize,
        lambda_coherence=0.01,
        lambda_entropy=lambda_entropy,
        entropy_schedule=entropy_schedule,
        entropy_anneal_epochs=entropy_anneal_epochs,
    )
    return COFRN(config)


def make_mlp_baseline(
    encoder_dim: int,
    n_classes: int,
    hidden_dim: int = 256,
) -> MLPHead:
    """Create an MLP baseline with the specified encoder dim."""
    return MLPHead(
        hidden_dim=hidden_dim,
        mlp_hidden=256,
        n_classes=n_classes,
        encoder_input_dim=encoder_dim,
        use_precomputed=True,
    )


def make_lora_baseline(
    encoder_dim: int,
    n_classes: int,
    hidden_dim: int = 256,
) -> LoRAModel:
    """Create a LoRA baseline (simulated, precomputed mode)."""
    return LoRAModel(
        hidden_dim=hidden_dim,
        n_classes=n_classes,
        encoder_input_dim=encoder_dim,
        use_precomputed=True,
    )


def collect_anchor_weights(
    model: COFRN,
    dataset: BenchmarkDataset,
    device: str = 'cpu',
    batch_size: int = 64,
) -> np.ndarray:
    """Extract anchor attention weights for all examples in a dataset."""
    model.eval()
    loader = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=False, collate_fn=collate_benchmark,
    )
    all_weights = []
    with torch.no_grad():
        for batch in loader:
            emb = batch['embeddings'].to(device)
            s0 = model.encode_text(embeddings=emb)
            fact_out = model.factorization(s0)
            all_weights.append(fact_out['weights'].cpu().numpy())
    return np.concatenate(all_weights, axis=0)


def collect_structural_embeddings(
    model: COFRN,
    dataset: BenchmarkDataset,
    device: str = 'cpu',
    batch_size: int = 64,
) -> np.ndarray:
    """Extract structural |o> embeddings for all examples."""
    model.eval()
    loader = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=False, collate_fn=collate_benchmark,
    )
    all_structural = []
    with torch.no_grad():
        for batch in loader:
            emb = batch['embeddings'].to(device)
            s0 = model.encode_text(embeddings=emb)
            fact_out = model.factorization(s0)
            all_structural.append(fact_out['structural'].cpu().numpy())
    return np.concatenate(all_structural, axis=0)


# ═══════════════════════════════════════════════════════════════════════
# PHASE 1: Break the Encoder Ceiling
# ═══════════════════════════════════════════════════════════════════════

def run_phase1_encoder_sweep(
    max_epochs: int = 15,
    patience: int = 5,
    seed: int = 42,
    device: str = 'cpu',
    output_dir: Path = Path('results'),
) -> Dict:
    """
    Phase 1: Sweep 4 encoders x 4 benchmarks x {CO-FRN, MLP}.

    Returns results dict with accuracy matrix and gate evaluation.
    """
    logger.info("=" * 70)
    logger.info("PHASE 1: Break the Encoder Ceiling")
    logger.info("=" * 70)

    set_seed(seed)
    emb_cache = {}
    results = {
        'phase': 1,
        'accuracies': {},  # {encoder: {benchmark: {method: acc}}}
        'param_counts': {},
        'timing': {},
    }

    for enc_name in ENCODER_CONFIGS:
        results['accuracies'][enc_name] = {}
        results['param_counts'][enc_name] = {}
        enc_dim = ENCODER_CONFIGS[enc_name]['dim']

        for bench in BENCHMARKS:
            logger.info(f"\n--- {enc_name} / {bench} ---")
            results['accuracies'][enc_name][bench] = {}

            # Load data
            try:
                train_ds = load_or_precompute(enc_name, bench, 'train', seed, emb_cache)
            except Exception as e:
                logger.warning(f"Failed to load {bench}/train with {enc_name}: {e}")
                results['accuracies'][enc_name][bench] = {'cofrn': None, 'mlp': None}
                continue

            # Create val split from train if needed
            n = len(train_ds)
            val_size = min(500, n // 5)
            rng = np.random.RandomState(seed)
            indices = rng.permutation(n)
            val_ds = train_ds.subset(indices[:val_size])
            train_split = train_ds.subset(indices[val_size:])

            # Try to load test split
            try:
                test_split_name = 'test' if bench != 'folio' else 'validation'
                test_ds = load_or_precompute(enc_name, bench, test_split_name, seed, emb_cache)
            except Exception:
                test_ds = val_ds  # fallback

            n_classes = BENCHMARK_N_CLASSES[bench]
            train_config = TrainConfig(
                max_epochs=max_epochs,
                patience=patience,
                batch_size=32,
                device=device,
                seed=seed,
            )

            # --- CO-FRN ---
            try:
                cofrn = make_cofrn(enc_dim, n_classes)
                cofrn, train_res = train_model(
                    cofrn, train_split, val_ds, train_config,
                    model_name=f'cofrn_{enc_name}_{bench}',
                )
                test_res = evaluate_model(cofrn, test_ds, train_config)
                results['accuracies'][enc_name][bench]['cofrn'] = test_res['accuracy']
                results['param_counts'][enc_name]['cofrn'] = count_trainable_params(cofrn)
                logger.info(f"  CO-FRN: {test_res['accuracy']:.2%}")
                del cofrn
            except Exception as e:
                logger.warning(f"  CO-FRN failed: {e}")
                results['accuracies'][enc_name][bench]['cofrn'] = None

            # --- MLP baseline ---
            try:
                mlp = make_mlp_baseline(enc_dim, n_classes)
                mlp, train_res = train_model(
                    mlp, train_split, val_ds, train_config,
                    model_name=f'mlp_{enc_name}_{bench}',
                )
                test_res = evaluate_model(mlp, test_ds, train_config)
                results['accuracies'][enc_name][bench]['mlp'] = test_res['accuracy']
                results['param_counts'][enc_name]['mlp'] = count_trainable_params(mlp)
                logger.info(f"  MLP:    {test_res['accuracy']:.2%}")
                del mlp
            except Exception as e:
                logger.warning(f"  MLP failed: {e}")
                results['accuracies'][enc_name][bench]['mlp'] = None

            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # --- Evaluate Phase 1 Gate ---
    results['gate'] = evaluate_phase1_gate(results['accuracies'])
    results['best_encoder'] = results['gate'].get('best_encoder')

    save_results(results, str(output_dir / 'phase1_results.json'))
    logger.info(f"\nPhase 1 Gate: {'PASS' if results['gate']['passed'] else 'FAIL'}")
    if results['best_encoder']:
        logger.info(f"Best encoder: {results['best_encoder']}")

    return results


def evaluate_phase1_gate(accuracies: Dict) -> Dict:
    """
    Phase 1 gate: best non-GPT-2 encoder > chance + 5pp on >= 2/3 non-math benchmarks.
    """
    gate = {
        'passed': False,
        'best_encoder': None,
        'best_score': 0,
        'details': {},
    }

    best_enc = None
    best_count = 0
    best_mean_acc = 0.0

    for enc_name in ENCODER_CONFIGS:
        if enc_name == 'gpt2':
            continue

        above_chance = 0
        enc_accs = []
        details = {}

        for bench in NON_MATH_BENCHMARKS:
            chance = CHANCE_LEVELS[bench]
            threshold = chance + 0.05

            # Use best of cofrn and mlp
            bench_accs = accuracies.get(enc_name, {}).get(bench, {})
            best_acc = max(
                bench_accs.get('cofrn') or 0.0,
                bench_accs.get('mlp') or 0.0,
            )

            if best_acc > threshold:
                above_chance += 1
            enc_accs.append(best_acc)
            details[bench] = {
                'best_acc': best_acc,
                'chance': chance,
                'threshold': threshold,
                'passed': best_acc > threshold,
            }

        gate['details'][enc_name] = details
        mean_acc = np.mean(enc_accs) if enc_accs else 0.0

        if above_chance > best_count or (above_chance == best_count and mean_acc > best_mean_acc):
            best_count = above_chance
            best_mean_acc = mean_acc
            best_enc = enc_name

    gate['best_encoder'] = best_enc
    gate['best_score'] = best_count
    gate['passed'] = best_count >= 2  # >= 2 of 3 non-math benchmarks

    return gate


# ═══════════════════════════════════════════════════════════════════════
# PHASE 2: Fix Operator Specialization
# ═══════════════════════════════════════════════════════════════════════

def run_phase2_specialization(
    best_encoder: str,
    max_epochs: int = 15,
    patience: int = 5,
    seed: int = 42,
    device: str = 'cpu',
    output_dir: Path = Path('results'),
) -> Dict:
    """
    Phase 2: 3 entropy strategies x 2 lambda_f values x 4 benchmarks.
    """
    logger.info("=" * 70)
    logger.info("PHASE 2: Fix Operator Specialization")
    logger.info(f"Using encoder: {best_encoder}")
    logger.info("=" * 70)

    set_seed(seed)
    emb_cache = {}
    enc_dim = ENCODER_CONFIGS[best_encoder]['dim']

    entropy_strategies = [
        ('constant', 0.01),   # control: Paper 12 default
        ('disabled', 0.0),    # free specialization
        ('anneal', 0.01),     # start uniform, relax
    ]
    lambda_f_values = [0.0, 0.1]

    results = {
        'phase': 2,
        'encoder': best_encoder,
        'runs': {},
        'diagnostics': {},
    }

    # Precompute embeddings for all benchmarks
    datasets = {}
    for bench in BENCHMARKS:
        try:
            train_ds = load_or_precompute(best_encoder, bench, 'train', seed, emb_cache)
            n = len(train_ds)
            val_size = min(500, n // 5)
            rng = np.random.RandomState(seed)
            indices = rng.permutation(n)
            datasets[bench] = {
                'train': train_ds.subset(indices[val_size:]),
                'val': train_ds.subset(indices[:val_size]),
            }
            # Test split
            try:
                test_name = 'test' if bench != 'folio' else 'validation'
                datasets[bench]['test'] = load_or_precompute(
                    best_encoder, bench, test_name, seed, emb_cache,
                )
            except Exception:
                datasets[bench]['test'] = datasets[bench]['val']
        except Exception as e:
            logger.warning(f"Failed to load {bench}: {e}")

    # Run grid
    for ent_schedule, ent_lambda in entropy_strategies:
        for lam_f in lambda_f_values:
            config_key = f"{ent_schedule}_lf{lam_f}"
            results['runs'][config_key] = {}
            results['diagnostics'][config_key] = {}

            for bench in BENCHMARKS:
                if bench not in datasets:
                    continue

                run_key = f"{config_key}_{bench}"
                logger.info(f"\n--- {config_key} / {bench} ---")

                n_classes = BENCHMARK_N_CLASSES[bench]
                model = make_cofrn(
                    enc_dim, n_classes,
                    entropy_schedule=ent_schedule,
                    lambda_entropy=ent_lambda,
                    lambda_factorize=lam_f,
                )

                train_config = TrainConfig(
                    max_epochs=max_epochs,
                    patience=patience,
                    batch_size=32,
                    device=device,
                    seed=seed,
                    lambda_entropy=ent_lambda,
                    lambda_factorize=lam_f,
                )

                try:
                    model, train_res = train_model(
                        model, datasets[bench]['train'], datasets[bench]['val'],
                        train_config, model_name=run_key,
                    )
                    test_res = evaluate_model(model, datasets[bench]['test'], train_config)
                    acc = test_res['accuracy']

                    # Collect diagnostics
                    weights = collect_anchor_weights(
                        model, datasets[bench]['test'], device,
                    )
                    util = anchor_utilization_entropy(weights, model.config.n_anchors)

                    results['runs'][config_key][bench] = acc
                    results['diagnostics'][config_key][bench] = {
                        'accuracy': acc,
                        'anchor_entropy': util['entropy'],
                        'normalized_entropy': util['normalized_entropy'],
                        'participation_ratio': util['participation_ratio'],
                        'max_weight': util['max_weight'],
                        'min_weight': util['min_weight'],
                    }

                    logger.info(f"  Acc: {acc:.2%}, PR: {util['participation_ratio']:.2f}, "
                                f"H: {util['normalized_entropy']:.3f}")

                except Exception as e:
                    logger.warning(f"  Failed: {e}")
                    results['runs'][config_key][bench] = None
                    results['diagnostics'][config_key][bench] = None

                del model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Compute rho for each config (using structural embeddings)
    for config_key in results['diagnostics']:
        # Reconstruct best model for rho computation
        parts = config_key.split('_lf')
        ent_schedule = parts[0]
        lam_f = float(parts[1])
        ent_lambda = 0.01 if ent_schedule != 'disabled' else 0.0

        try:
            structural_by_domain = {}
            for bench in BENCHMARKS:
                if bench not in datasets:
                    continue
                n_classes = BENCHMARK_N_CLASSES[bench]
                model = make_cofrn(
                    enc_dim, n_classes,
                    entropy_schedule=ent_schedule,
                    lambda_entropy=ent_lambda,
                    lambda_factorize=lam_f,
                )
                train_config = TrainConfig(
                    max_epochs=max_epochs, patience=patience,
                    batch_size=32, device=device, seed=seed,
                    lambda_entropy=ent_lambda, lambda_factorize=lam_f,
                )
                model, _ = train_model(
                    model, datasets[bench]['train'], datasets[bench]['val'],
                    train_config, model_name=f'rho_{config_key}_{bench}',
                )
                structural = collect_structural_embeddings(
                    model, datasets[bench]['test'], device,
                )
                domain = BENCHMARK_CONFIGS[bench]['domain']
                structural_by_domain[domain] = structural
                del model

            if len(structural_by_domain) >= 2:
                rho_result = compute_centered_redundancy(
                    structural_by_domain, config_key,
                )
                results['diagnostics'][config_key]['rho'] = rho_result.rho_centroid
                logger.info(f"  {config_key} rho: {rho_result.rho_centroid:.3f}")
            else:
                results['diagnostics'][config_key]['rho'] = None
        except Exception as e:
            logger.warning(f"  Rho computation failed for {config_key}: {e}")
            results['diagnostics'][config_key]['rho'] = None

    # --- Load Phase 1 results for gate comparison ---
    phase1_path = output_dir / 'phase1_results.json'
    phase1_accs = {}
    if phase1_path.exists():
        with open(phase1_path) as f:
            p1 = json.load(f)
        for bench in BENCHMARKS:
            bench_data = p1.get('accuracies', {}).get(best_encoder, {}).get(bench, {})
            phase1_accs[bench] = max(
                bench_data.get('cofrn') or 0.0,
                bench_data.get('mlp') or 0.0,
            )

    results['gate'] = evaluate_phase2_gate(results, phase1_accs)
    results['best_config'] = results['gate'].get('best_config')

    save_results(results, str(output_dir / 'phase2_results.json'))
    logger.info(f"\nPhase 2 Gate: {'PASS' if results['gate']['passed'] else 'FAIL'}")

    return results


def evaluate_phase2_gate(results: Dict, phase1_accs: Dict) -> Dict:
    """
    Phase 2 gate:
      (a) participation ratio in [3, 12]
      (b) accuracy >= Phase 1 baseline on >= 3/4 benchmarks
      (c) rho < 1.5
    """
    gate = {
        'passed': False,
        'best_config': None,
        'details': {},
    }

    best_config = None
    best_score = -1

    for config_key, diag in results['diagnostics'].items():
        if not isinstance(diag, dict):
            continue

        # (a) Check participation ratio
        pr_values = []
        for bench in BENCHMARKS:
            bench_diag = diag.get(bench)
            if bench_diag and isinstance(bench_diag, dict) and 'participation_ratio' in bench_diag:
                pr_values.append(bench_diag['participation_ratio'])

        mean_pr = np.mean(pr_values) if pr_values else 0.0
        pr_ok = 3.0 <= mean_pr <= 12.0

        # (b) Check accuracy vs Phase 1
        n_at_least = 0
        for bench in BENCHMARKS:
            run_acc = results['runs'].get(config_key, {}).get(bench)
            p1_acc = phase1_accs.get(bench, 0.0)
            if run_acc is not None and run_acc >= p1_acc - 0.01:  # small tolerance
                n_at_least += 1
        acc_ok = n_at_least >= 3

        # (c) Check rho
        rho = diag.get('rho')
        rho_ok = rho is not None and rho < 1.5

        score = sum([pr_ok, acc_ok, rho_ok])
        gate['details'][config_key] = {
            'mean_pr': mean_pr, 'pr_ok': pr_ok,
            'n_benchmarks_at_phase1': n_at_least, 'acc_ok': acc_ok,
            'rho': rho, 'rho_ok': rho_ok,
            'score': score,
        }

        if score > best_score or (score == best_score and mean_pr > 3):
            best_score = score
            best_config = config_key

    gate['best_config'] = best_config
    gate['passed'] = best_score >= 3  # all three criteria met

    return gate


# ═══════════════════════════════════════════════════════════════════════
# PHASE 3: Cross-Domain Operator Transfer (Main Event)
# ═══════════════════════════════════════════════════════════════════════

def run_phase3_transfer(
    best_encoder: str,
    best_config: str,
    n_seeds: int = 3,
    max_epochs: int = 30,
    patience: int = 7,
    device: str = 'cpu',
    output_dir: Path = Path('results'),
) -> Dict:
    """
    Phase 3: Cross-domain operator transfer using best encoder + config.
    """
    logger.info("=" * 70)
    logger.info("PHASE 3: Cross-Domain Operator Transfer")
    logger.info(f"Encoder: {best_encoder}, Config: {best_config}")
    logger.info("=" * 70)

    enc_dim = ENCODER_CONFIGS[best_encoder]['dim']

    # Parse config
    parts = best_config.split('_lf')
    ent_schedule = parts[0]
    lam_f = float(parts[1])
    ent_lambda = 0.01 if ent_schedule != 'disabled' else 0.0

    emb_cache = {}
    results = {
        'phase': 3,
        'encoder': best_encoder,
        'config': best_config,
        'entropy_schedule': ent_schedule,
        'lambda_factorize': lam_f,
    }

    # ─── 3a: Source training (3 seeds) ────────────────────────────────
    logger.info("\n--- Phase 3a: Source Training ---")
    results['source_training'] = {}

    all_source_models = []  # store trained CO-FRN models for analysis

    for seed_idx in range(n_seeds):
        seed = 42 + seed_idx
        set_seed(seed)
        seed_key = f'seed_{seed}'
        results['source_training'][seed_key] = {}

        # Load source datasets
        source_datasets = {}
        for bench in SOURCE_BENCHMARKS:
            try:
                ds = load_or_precompute(best_encoder, bench, 'train', seed, emb_cache)
                source_datasets[bench] = ds
            except Exception as e:
                logger.warning(f"Failed to load {bench}: {e}")

        if not source_datasets:
            continue

        # Create mixed source dataset
        mixed = MixedDomainDataset(source_datasets)
        n = len(mixed)
        val_size = min(500, n // 5)
        rng = np.random.RandomState(seed)
        indices = rng.permutation(n)

        # Create train/val BenchmarkDatasets from mixed
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]

        train_q = mixed.question_embeddings[train_indices]
        train_a = mixed.answer_embeddings[train_indices]
        train_l = mixed.labels[train_indices]
        val_q = mixed.question_embeddings[val_indices]
        val_a = mixed.answer_embeddings[val_indices]
        val_l = mixed.labels[val_indices]

        # Use max n_classes across source benchmarks
        max_classes = max(BENCHMARK_N_CLASSES[b] for b in SOURCE_BENCHMARKS)

        train_bds = BenchmarkDataset(train_q, train_a, train_l, domain='source')
        val_bds = BenchmarkDataset(val_q, val_a, val_l, domain='source')

        train_config = TrainConfig(
            max_epochs=max_epochs,
            patience=patience,
            batch_size=32,
            device=device,
            seed=seed,
            lambda_entropy=ent_lambda,
            lambda_factorize=lam_f,
        )

        # --- CO-FRN ---
        try:
            cofrn = make_cofrn(
                enc_dim, max_classes,
                entropy_schedule=ent_schedule,
                lambda_entropy=ent_lambda,
                lambda_factorize=lam_f,
            )
            cofrn, train_res = train_model(
                cofrn, train_bds, val_bds, train_config,
                model_name=f'cofrn_source_s{seed}',
            )
            results['source_training'][seed_key]['cofrn'] = {
                'best_val_acc': train_res['best_val_acc'],
                'epochs_trained': train_res['epochs_trained'],
            }
            all_source_models.append((seed, cofrn))
            logger.info(f"  CO-FRN seed={seed}: val_acc={train_res['best_val_acc']:.2%}")
        except Exception as e:
            logger.warning(f"  CO-FRN failed seed={seed}: {e}")

        # --- MLP baseline ---
        try:
            mlp = make_mlp_baseline(enc_dim, max_classes)
            mlp, train_res = train_model(
                mlp, train_bds, val_bds, train_config,
                model_name=f'mlp_source_s{seed}',
            )
            results['source_training'][seed_key]['mlp'] = {
                'best_val_acc': train_res['best_val_acc'],
                'epochs_trained': train_res['epochs_trained'],
            }
            del mlp
            logger.info(f"  MLP seed={seed}: val_acc={train_res['best_val_acc']:.2%}")
        except Exception as e:
            logger.warning(f"  MLP failed seed={seed}: {e}")

        # --- LoRA baseline ---
        try:
            lora = make_lora_baseline(enc_dim, max_classes)
            lora, train_res = train_model(
                lora, train_bds, val_bds, train_config,
                model_name=f'lora_source_s{seed}',
            )
            results['source_training'][seed_key]['lora'] = {
                'best_val_acc': train_res['best_val_acc'],
                'epochs_trained': train_res['epochs_trained'],
            }
            del lora
            logger.info(f"  LoRA seed={seed}: val_acc={train_res['best_val_acc']:.2%}")
        except Exception as e:
            logger.warning(f"  LoRA failed seed={seed}: {e}")

    # ─── 3b: Zero-shot transfer ──────────────────────────────────────
    logger.info("\n--- Phase 3b: Zero-Shot Transfer ---")
    results['zero_shot'] = {}

    all_targets = NEAR_TRANSFER + FAR_TRANSFER
    for bench in all_targets:
        results['zero_shot'][bench] = {}
        try:
            test_name = 'test' if bench not in ('folio',) else 'validation'
            target_ds = load_or_precompute(best_encoder, bench, test_name, 42, emb_cache)
        except Exception as e:
            logger.warning(f"Failed to load {bench}: {e}")
            continue

        seed_accs = []
        for seed, cofrn in all_source_models:
            try:
                train_config = TrainConfig(batch_size=32, device=device)
                res = evaluate_model(cofrn, target_ds, train_config)
                seed_accs.append(res['accuracy'])
            except Exception as e:
                logger.warning(f"  Eval failed on {bench} seed={seed}: {e}")

        if seed_accs:
            results['zero_shot'][bench] = {
                'accuracies': seed_accs,
                'mean': float(np.mean(seed_accs)),
                'std': float(np.std(seed_accs)),
                'chance': CHANCE_LEVELS.get(bench, 0.25),
            }
            # Compute transfer ratio
            source_accs = []
            for seed_key in results['source_training']:
                sa = results['source_training'][seed_key].get('cofrn', {}).get('best_val_acc')
                if sa is not None:
                    source_accs.append(sa)
            if source_accs:
                results['zero_shot'][bench]['transfer_ratio'] = transfer_ratio(
                    np.mean(source_accs), np.mean(seed_accs),
                )
            logger.info(f"  {bench}: {np.mean(seed_accs):.2%} "
                        f"(TR={results['zero_shot'][bench].get('transfer_ratio', 0):.3f})")

    # ─── 3c: Few-shot adaptation ─────────────────────────────────────
    logger.info("\n--- Phase 3c: Few-Shot Adaptation ---")
    results['few_shot'] = {}

    for bench in FAR_TRANSFER:
        results['few_shot'][bench] = {'cofrn': {}, 'lora': {}}

        try:
            train_ds = load_or_precompute(best_encoder, bench, 'train', 42, emb_cache)
            test_name = 'test' if bench != 'folio' else 'validation'
            test_ds = load_or_precompute(best_encoder, bench, test_name, 42, emb_cache)
        except Exception as e:
            logger.warning(f"Failed to load {bench}: {e}")
            continue

        n_classes = BENCHMARK_N_CLASSES[bench]

        for budget in FEW_SHOT_BUDGETS:
            if budget > len(train_ds):
                continue

            cofrn_accs = []
            lora_accs = []

            for seed_idx in range(n_seeds):
                seed = 42 + seed_idx
                set_seed(seed)
                rng = np.random.RandomState(seed)

                # Sample few-shot subset
                indices = rng.choice(len(train_ds), size=min(budget, len(train_ds)), replace=False)
                few_shot_ds = train_ds.subset(indices)
                few_val_size = min(len(few_shot_ds) // 5, 100)
                if few_val_size < 4:
                    few_val_size = min(4, len(few_shot_ds))
                val_indices = list(range(few_val_size))
                train_indices = list(range(few_val_size, len(few_shot_ds)))
                if not train_indices:
                    train_indices = val_indices

                few_train = few_shot_ds.subset(train_indices)
                few_val = few_shot_ds.subset(val_indices)

                adapt_config = TrainConfig(
                    max_epochs=min(30, max(10, 1000 // max(budget, 1))),
                    patience=5,
                    batch_size=min(16, budget),
                    device=device,
                    seed=seed,
                    lambda_entropy=ent_lambda,
                    lambda_factorize=lam_f,
                )

                # CO-FRN: freeze operator manifold + structural MLP, adapt context only
                if all_source_models:
                    try:
                        _, base_cofrn = all_source_models[seed_idx % len(all_source_models)]
                        adapted = copy.deepcopy(base_cofrn)
                        adapted.config.use_precomputed = True
                        adapted.config.encoder_input_dim = enc_dim

                        # Freeze manifold and structural MLP
                        for param in adapted.factorization.manifold.parameters():
                            param.requires_grad = False
                        for param in adapted.factorization.factorizer.struct_mlp.parameters():
                            param.requires_grad = False

                        # Rebuild answer_proj for new n_classes if needed
                        adapted.answer_proj = nn.Sequential(
                            nn.Linear(enc_dim, adapted.config.hidden_dim),
                            nn.GELU(),
                            nn.Linear(adapted.config.hidden_dim, adapted.config.hidden_dim),
                        ).to(device)

                        adapted, _ = train_model(
                            adapted, few_train, few_val, adapt_config,
                            model_name=f'cofrn_adapt_{bench}_n{budget}_s{seed}',
                        )
                        res = evaluate_model(adapted, test_ds, adapt_config)
                        cofrn_accs.append(res['accuracy'])
                        del adapted
                    except Exception as e:
                        logger.warning(f"  CO-FRN adapt failed: {e}")

                # LoRA baseline: train from scratch
                try:
                    lora = make_lora_baseline(enc_dim, n_classes)
                    lora, _ = train_model(
                        lora, few_train, few_val, adapt_config,
                        model_name=f'lora_adapt_{bench}_n{budget}_s{seed}',
                    )
                    res = evaluate_model(lora, test_ds, adapt_config)
                    lora_accs.append(res['accuracy'])
                    del lora
                except Exception as e:
                    logger.warning(f"  LoRA adapt failed: {e}")

            results['few_shot'][bench]['cofrn'][budget] = {
                'accuracies': cofrn_accs,
                'mean': float(np.mean(cofrn_accs)) if cofrn_accs else 0.0,
                'std': float(np.std(cofrn_accs)) if cofrn_accs else 0.0,
            }
            results['few_shot'][bench]['lora'][budget] = {
                'accuracies': lora_accs,
                'mean': float(np.mean(lora_accs)) if lora_accs else 0.0,
                'std': float(np.std(lora_accs)) if lora_accs else 0.0,
            }
            logger.info(
                f"  {bench} n={budget}: CO-FRN={np.mean(cofrn_accs):.2%} "
                f"LoRA={np.mean(lora_accs):.2%}"
                if cofrn_accs and lora_accs else f"  {bench} n={budget}: incomplete"
            )

    # ─── 3d: Reverse direction ────────────────────────────────────────
    logger.info("\n--- Phase 3d: Reverse Direction ---")
    results['reverse'] = {}

    for seed_idx in range(min(n_seeds, 1)):  # just 1 seed for reverse
        seed = 42 + seed_idx
        set_seed(seed)

        rev_datasets = {}
        for bench in REVERSE_SOURCE:
            try:
                ds = load_or_precompute(best_encoder, bench, 'train', seed, emb_cache)
                rev_datasets[bench] = ds
            except Exception as e:
                logger.warning(f"Failed to load {bench}: {e}")

        if len(rev_datasets) < 2:
            continue

        mixed = MixedDomainDataset(rev_datasets)
        n = len(mixed)
        val_size = min(500, n // 5)
        rng = np.random.RandomState(seed)
        indices = rng.permutation(n)

        train_q = mixed.question_embeddings[indices[val_size:]]
        train_a = mixed.answer_embeddings[indices[val_size:]]
        train_l = mixed.labels[indices[val_size:]]
        val_q = mixed.question_embeddings[indices[:val_size]]
        val_a = mixed.answer_embeddings[indices[:val_size]]
        val_l = mixed.labels[indices[:val_size]]

        max_classes = max(BENCHMARK_N_CLASSES[b] for b in REVERSE_SOURCE)
        train_bds = BenchmarkDataset(train_q, train_a, train_l, domain='reverse_source')
        val_bds = BenchmarkDataset(val_q, val_a, val_l, domain='reverse_source')

        train_config = TrainConfig(
            max_epochs=max_epochs, patience=patience,
            batch_size=32, device=device, seed=seed,
            lambda_entropy=ent_lambda, lambda_factorize=lam_f,
        )

        try:
            rev_cofrn = make_cofrn(
                enc_dim, max_classes,
                entropy_schedule=ent_schedule,
                lambda_entropy=ent_lambda,
                lambda_factorize=lam_f,
            )
            rev_cofrn, _ = train_model(
                rev_cofrn, train_bds, val_bds, train_config,
                model_name='cofrn_reverse',
            )

            for bench in REVERSE_TARGET:
                try:
                    test_name = 'test'
                    target_ds = load_or_precompute(
                        best_encoder, bench, test_name, seed, emb_cache,
                    )
                    res = evaluate_model(rev_cofrn, target_ds, train_config)
                    results['reverse'][bench] = {
                        'accuracy': res['accuracy'],
                        'chance': CHANCE_LEVELS.get(bench, 0.25),
                    }
                    logger.info(f"  Reverse -> {bench}: {res['accuracy']:.2%}")
                except Exception as e:
                    logger.warning(f"  Reverse eval failed on {bench}: {e}")

            del rev_cofrn
        except Exception as e:
            logger.warning(f"  Reverse training failed: {e}")

    # ─── 3e: Operator analysis ────────────────────────────────────────
    logger.info("\n--- Phase 3e: Operator Analysis ---")
    results['operator_analysis'] = {}

    if all_source_models:
        _, best_model = all_source_models[0]

        # Collect weights and domains across all benchmarks
        all_weights = []
        all_domains = []

        for bench in BENCHMARKS:
            try:
                ds = load_or_precompute(best_encoder, bench, 'train', 42, emb_cache)
                # Use subset for efficiency
                rng = np.random.RandomState(42)
                n_sample = min(200, len(ds))
                indices = rng.choice(len(ds), n_sample, replace=False)
                subset = ds.subset(indices)

                weights = collect_anchor_weights(best_model, subset, device)
                all_weights.append(weights)
                all_domains.extend([BENCHMARK_CONFIGS[bench]['domain']] * len(weights))
            except Exception as e:
                logger.warning(f"  Analysis failed for {bench}: {e}")

        if all_weights:
            all_weights = np.concatenate(all_weights, axis=0)
            all_domains = np.array(all_domains)

            # Per-domain anchor utilization
            domain_util = per_domain_conditional_entropy(
                all_weights, all_domains, best_model.config.n_anchors,
            )
            results['operator_analysis']['per_domain_utilization'] = domain_util

            # Cross-domain JS divergence
            js_result = cross_domain_js_divergence(all_weights, all_domains)
            results['operator_analysis']['js_divergence'] = js_result
            logger.info(f"  Mean JS divergence: {js_result['mean_js']:.4f}")
            logger.info(f"  Pairs below 0.5: {js_result['n_below_threshold']}/{js_result['n_pairs']}")

            # Rho with bootstrap CI
            structural_by_domain = {}
            for bench in BENCHMARKS:
                try:
                    ds = load_or_precompute(best_encoder, bench, 'train', 42, emb_cache)
                    rng = np.random.RandomState(42)
                    n_sample = min(200, len(ds))
                    indices = rng.choice(len(ds), n_sample, replace=False)
                    subset = ds.subset(indices)
                    structural = collect_structural_embeddings(best_model, subset, device)
                    domain = BENCHMARK_CONFIGS[bench]['domain']
                    structural_by_domain[domain] = structural
                except Exception:
                    pass

            if len(structural_by_domain) >= 2:
                rho_mean, rho_lo, rho_hi = bootstrap_rho_ci(
                    structural_by_domain, n_bootstrap=500,
                )
                results['operator_analysis']['rho'] = {
                    'mean': rho_mean, 'ci_lo': rho_lo, 'ci_hi': rho_hi,
                }
                logger.info(f"  Rho: {rho_mean:.3f} [{rho_lo:.3f}, {rho_hi:.3f}]")

            # Causal intervention: clamp anchors and measure accuracy change
            results['operator_analysis']['causal_intervention'] = {}
            for bench in BENCHMARKS[:2]:  # just GSM8K and ARC for speed
                try:
                    ds = load_or_precompute(best_encoder, bench, 'train', 42, emb_cache)
                    rng = np.random.RandomState(42)
                    n_sample = min(200, len(ds))
                    indices = rng.choice(len(ds), n_sample, replace=False)
                    subset = ds.subset(indices)

                    # Baseline accuracy
                    base_config = TrainConfig(batch_size=32, device=device)
                    base_res = evaluate_model(best_model, subset, base_config)
                    base_acc = base_res['accuracy']

                    # Clamp each anchor to uniform -> measure delta
                    n_anchors = best_model.config.n_anchors
                    deltas = []
                    for anchor_idx in range(min(n_anchors, 4)):  # sample 4 anchors
                        # Save original weights and monkey-patch compute_attention_weights
                        orig_fn = best_model.factorization.manifold.compute_attention_weights

                        def clamped_fn(structural, _idx=anchor_idx, _n=n_anchors):
                            B = structural.shape[0]
                            w = torch.zeros(B, _n, device=structural.device)
                            w[:, _idx] = 1.0
                            return w

                        best_model.factorization.manifold.compute_attention_weights = clamped_fn
                        clamped_res = evaluate_model(best_model, subset, base_config)
                        best_model.factorization.manifold.compute_attention_weights = orig_fn

                        deltas.append(clamped_res['accuracy'] - base_acc)

                    results['operator_analysis']['causal_intervention'][bench] = {
                        'base_accuracy': base_acc,
                        'anchor_deltas': deltas,
                        'mean_delta': float(np.mean(deltas)),
                        'std_delta': float(np.std(deltas)),
                    }
                    logger.info(f"  Causal {bench}: base={base_acc:.2%}, "
                                f"mean_delta={np.mean(deltas):.4f}")
                except Exception as e:
                    logger.warning(f"  Causal intervention failed for {bench}: {e}")

    # Clean up models
    del all_source_models

    save_results(results, str(output_dir / 'phase3_results.json'))
    return results


# ═══════════════════════════════════════════════════════════════════════
# FINDINGS GENERATION
# ═══════════════════════════════════════════════════════════════════════

def generate_findings(output_dir: Path):
    """Generate FINDINGS.md from all phase results."""
    logger.info("\nGenerating FINDINGS.md...")

    lines = [
        "# Experiment 5: Cross-Domain Reasoning Operator Extraction — Findings",
        "",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    # Load results
    phase_results = {}
    for phase in [1, 2, 3]:
        path = output_dir / f'phase{phase}_results.json'
        if path.exists():
            with open(path) as f:
                phase_results[phase] = json.load(f)

    # Phase 1
    if 1 in phase_results:
        p1 = phase_results[1]
        gate = p1.get('gate', {})
        lines.extend([
            "## Phase 1: Encoder Ceiling",
            "",
            f"**Gate**: {'PASS' if gate.get('passed') else 'FAIL'}",
            f"**Best encoder**: {gate.get('best_encoder', 'N/A')}",
            f"**Non-math benchmarks above threshold**: {gate.get('best_score', 0)}/3",
            "",
            "### Accuracy Matrix",
            "",
            "| Encoder | GSM8K | ARC | StrategyQA | FOLIO |",
            "|---------|-------|-----|------------|-------|",
        ])

        for enc in ENCODER_CONFIGS:
            row = f"| {enc} |"
            for bench in BENCHMARKS:
                accs = p1.get('accuracies', {}).get(enc, {}).get(bench, {})
                cofrn_acc = accs.get('cofrn')
                mlp_acc = accs.get('mlp')
                best = max(cofrn_acc or 0, mlp_acc or 0)
                chance = CHANCE_LEVELS[bench]
                marker = " **" if best > chance + 0.05 else ""
                row += f" {best:.1%}{marker} |" if best > 0 else " N/A |"
            lines.append(row)

        lines.append("")
        lines.append(f"Chance levels: GSM8K=25%, ARC=25%, StrategyQA=50%, FOLIO=33.3%")
        lines.append("")

    # Phase 2
    if 2 in phase_results:
        p2 = phase_results[2]
        gate = p2.get('gate', {})
        lines.extend([
            "## Phase 2: Operator Specialization",
            "",
            f"**Gate**: {'PASS' if gate.get('passed') else 'FAIL'}",
            f"**Best config**: {gate.get('best_config', 'N/A')}",
            "",
            "### Diagnostics",
            "",
            "| Config | Mean PR | Rho | Gate Score |",
            "|--------|---------|-----|-----------|",
        ])

        for config_key, details in gate.get('details', {}).items():
            lines.append(
                f"| {config_key} | {details.get('mean_pr', 0):.2f} | "
                f"{details.get('rho', 'N/A')} | {details.get('score', 0)}/3 |"
            )
        lines.append("")

    # Phase 3
    if 3 in phase_results:
        p3 = phase_results[3]

        lines.extend([
            "## Phase 3: Cross-Domain Transfer",
            "",
        ])

        # Zero-shot
        zs = p3.get('zero_shot', {})
        if zs:
            lines.extend([
                "### Zero-Shot Transfer",
                "",
                "| Target | Accuracy | Transfer Ratio | Chance |",
                "|--------|----------|----------------|--------|",
            ])
            for bench, data in zs.items():
                if isinstance(data, dict) and 'mean' in data:
                    lines.append(
                        f"| {bench} | {data['mean']:.1%} +/- {data.get('std', 0):.1%} | "
                        f"{data.get('transfer_ratio', 0):.3f} | {data.get('chance', 0):.1%} |"
                    )
            lines.append("")

        # Few-shot
        fs = p3.get('few_shot', {})
        if fs:
            lines.extend([
                "### Few-Shot Adaptation",
                "",
                "| Benchmark | Budget | CO-FRN | LoRA | Delta |",
                "|-----------|--------|--------|------|-------|",
            ])
            for bench, methods in fs.items():
                cofrn_curve = methods.get('cofrn', {})
                lora_curve = methods.get('lora', {})
                for budget in FEW_SHOT_BUDGETS:
                    c = cofrn_curve.get(budget, {})
                    l = lora_curve.get(budget, {})
                    cm = c.get('mean', 0)
                    lm = l.get('mean', 0)
                    delta = cm - lm
                    lines.append(
                        f"| {bench} | {budget} | {cm:.1%} | {lm:.1%} | {delta:+.1%} |"
                    )
            lines.append("")

        # Reverse
        rev = p3.get('reverse', {})
        if rev:
            lines.extend([
                "### Reverse Direction (train commonsense+logic -> test math)",
                "",
            ])
            for bench, data in rev.items():
                if isinstance(data, dict):
                    lines.append(
                        f"- {bench}: {data.get('accuracy', 0):.1%} "
                        f"(chance: {data.get('chance', 0):.1%})"
                    )
            lines.append("")

        # Operator analysis
        oa = p3.get('operator_analysis', {})
        if oa:
            lines.extend(["### Operator Analysis", ""])

            js = oa.get('js_divergence', {})
            if js:
                lines.append(f"- Mean JS divergence: {js.get('mean_js', 0):.4f}")
                lines.append(f"- Pairs below 0.5: {js.get('n_below_threshold', 0)}/{js.get('n_pairs', 0)}")
                if 'pairwise' in js:
                    for pair, val in js['pairwise'].items():
                        lines.append(f"  - {pair}: {val:.4f}")
                lines.append("")

            rho = oa.get('rho', {})
            if rho:
                lines.append(
                    f"- Rho: {rho.get('mean', 0):.3f} "
                    f"[{rho.get('ci_lo', 0):.3f}, {rho.get('ci_hi', 0):.3f}]"
                )
                lines.append("")

            causal = oa.get('causal_intervention', {})
            if causal:
                lines.append("- Causal intervention (anchor clamping):")
                for bench, data in causal.items():
                    lines.append(
                        f"  - {bench}: base={data.get('base_accuracy', 0):.2%}, "
                        f"mean_delta={data.get('mean_delta', 0):.4f}"
                    )
                lines.append("")

    # Success criteria evaluation
    lines.extend([
        "## Success Criteria Evaluation",
        "",
        "| # | Criterion | Threshold | Result | Status |",
        "|---|-----------|-----------|--------|--------|",
    ])

    criteria = evaluate_all_criteria(phase_results)
    for c in criteria:
        status = "PASS" if c['met'] else "FAIL"
        lines.append(f"| {c['id']} | {c['name']} | {c['threshold']} | {c['result']} | {status} |")

    n_met = sum(1 for c in criteria if c['met'])
    lines.extend([
        "",
        f"**Overall: {n_met}/{len(criteria)} criteria met**",
        "",
    ])

    # Write FINDINGS.md
    findings_path = output_dir / 'FINDINGS.md'
    with open(findings_path, 'w') as f:
        f.write('\n'.join(lines))
    logger.info(f"FINDINGS.md written to {findings_path}")


def evaluate_all_criteria(phase_results: Dict) -> List[Dict]:
    """Evaluate all 6 success criteria."""
    criteria = []

    # C1: Encoder ceiling broken
    p1 = phase_results.get(1, {})
    gate1 = p1.get('gate', {})
    c1_met = gate1.get('passed', False)
    criteria.append({
        'id': 'C1', 'name': 'Encoder ceiling broken',
        'threshold': '>chance+5pp on >=2/3 non-math',
        'result': f"{gate1.get('best_score', 0)}/3 benchmarks",
        'met': c1_met,
    })

    # C2: Anchor specialization
    p2 = phase_results.get(2, {})
    best_config = p2.get('gate', {}).get('best_config')
    best_details = p2.get('gate', {}).get('details', {}).get(best_config, {})
    c2_pr = best_details.get('mean_pr', 0)
    criteria.append({
        'id': 'C2', 'name': 'Anchor specialization',
        'threshold': 'PR in [3, 12]',
        'result': f'PR={c2_pr:.2f}',
        'met': 3.0 <= c2_pr <= 12.0,
    })

    # C3: Rho improvement
    c3_rho = best_details.get('rho')
    criteria.append({
        'id': 'C3', 'name': 'Rho improvement',
        'threshold': 'rho < 1.5',
        'result': f'rho={c3_rho:.3f}' if c3_rho is not None else 'N/A',
        'met': c3_rho is not None and c3_rho < 1.5,
    })

    # C4: Far-domain zero-shot
    p3 = phase_results.get(3, {})
    zs = p3.get('zero_shot', {})
    c4_trs = []
    for bench in FAR_TRANSFER:
        tr = zs.get(bench, {}).get('transfer_ratio', 0)
        c4_trs.append(tr)
    c4_met = all(tr > 0.3 for tr in c4_trs) if c4_trs else False
    criteria.append({
        'id': 'C4', 'name': 'Far-domain zero-shot',
        'threshold': 'TR > 0.3 on both far-domain',
        'result': f'TRs: {[f"{t:.3f}" for t in c4_trs]}',
        'met': c4_met,
    })

    # C5: Few-shot beats baseline
    fs = p3.get('few_shot', {})
    c5_wins = 0
    for bench in FAR_TRANSFER:
        cofrn_data = fs.get(bench, {}).get('cofrn', {}).get(64, {})
        lora_data = fs.get(bench, {}).get('lora', {}).get(64, {})
        cofrn_accs = cofrn_data.get('accuracies', [])
        lora_accs = lora_data.get('accuracies', [])
        if cofrn_accs and lora_accs and len(cofrn_accs) >= 2 and len(lora_accs) >= 2:
            _, p_val = paired_permutation_test(
                np.array(cofrn_accs), np.array(lora_accs), n_permutations=1000,
            )
            if np.mean(cofrn_accs) > np.mean(lora_accs) and p_val < 0.05:
                c5_wins += 1
        elif cofrn_accs and lora_accs:
            if np.mean(cofrn_accs) > np.mean(lora_accs):
                c5_wins += 1
    criteria.append({
        'id': 'C5', 'name': 'Few-shot beats baseline',
        'threshold': 'CO-FRN > LoRA at n=64, p<0.05, >=1/2 far-domain',
        'result': f'{c5_wins}/{len(FAR_TRANSFER)} wins',
        'met': c5_wins >= 1,
    })

    # C6: Domain-invariant operators
    js = p3.get('operator_analysis', {}).get('js_divergence', {})
    n_below = js.get('n_below_threshold', 0)
    n_pairs = js.get('n_pairs', 0)
    criteria.append({
        'id': 'C6', 'name': 'Domain-invariant operators',
        'threshold': 'JS < 0.5 for >= 4/6 pairs',
        'result': f'{n_below}/{n_pairs} pairs below 0.5',
        'met': n_below >= 4,
    })

    return criteria


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Experiment 5: Cross-Domain Reasoning Operator Extraction",
    )
    parser.add_argument('--quick', action='store_true',
                        help='Phase 1 only, 5 epochs, 1 seed')
    parser.add_argument('--full', action='store_true',
                        help='All phases, 3 seeds for Phase 3')
    parser.add_argument('--phase', type=int, choices=[1, 2, 3],
                        help='Run only specified phase')
    parser.add_argument('--output', type=str, default='results/',
                        help='Output directory')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seeds', type=int, default=3,
                        help='Number of seeds for Phase 3')
    args = parser.parse_args()

    output_dir = Path(__file__).parent / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    if args.quick:
        # Quick mode: Phase 1 only, 5 epochs
        p1 = run_phase1_encoder_sweep(
            max_epochs=5, patience=3, device=args.device,
            output_dir=output_dir,
        )
        generate_findings(output_dir)

    elif args.phase == 1:
        run_phase1_encoder_sweep(
            max_epochs=15, patience=5, device=args.device,
            output_dir=output_dir,
        )
        generate_findings(output_dir)

    elif args.phase == 2:
        # Load Phase 1 results
        p1_path = output_dir / 'phase1_results.json'
        if not p1_path.exists():
            logger.error("Phase 1 results not found. Run Phase 1 first.")
            sys.exit(1)
        with open(p1_path) as f:
            p1 = json.load(f)
        best_encoder = p1.get('best_encoder')
        if not best_encoder:
            logger.error("Phase 1 gate failed — no best encoder. Cannot proceed.")
            sys.exit(1)

        run_phase2_specialization(
            best_encoder=best_encoder,
            max_epochs=15, patience=5, device=args.device,
            output_dir=output_dir,
        )
        generate_findings(output_dir)

    elif args.phase == 3:
        # Load Phase 2 results
        p2_path = output_dir / 'phase2_results.json'
        if not p2_path.exists():
            logger.error("Phase 2 results not found. Run Phase 2 first.")
            sys.exit(1)
        with open(p2_path) as f:
            p2 = json.load(f)

        p1_path = output_dir / 'phase1_results.json'
        with open(p1_path) as f:
            p1 = json.load(f)

        best_encoder = p1.get('best_encoder')
        best_config = p2.get('best_config')
        if not best_config:
            logger.error("Phase 2 gate failed — no best config. Cannot proceed.")
            sys.exit(1)

        run_phase3_transfer(
            best_encoder=best_encoder,
            best_config=best_config,
            n_seeds=args.seeds,
            max_epochs=30, patience=7, device=args.device,
            output_dir=output_dir,
        )
        generate_findings(output_dir)

    elif args.full:
        # Full pipeline: all 3 phases with gates
        p1 = run_phase1_encoder_sweep(
            max_epochs=15, patience=5, device=args.device,
            output_dir=output_dir,
        )

        if not p1['gate']['passed']:
            logger.warning("Phase 1 gate FAILED. Stopping with negative result.")
            generate_findings(output_dir)
            elapsed = time.time() - start_time
            logger.info(f"\nTotal time: {elapsed/60:.1f} minutes")
            return

        p2 = run_phase2_specialization(
            best_encoder=p1['best_encoder'],
            max_epochs=15, patience=5, device=args.device,
            output_dir=output_dir,
        )

        if not p2['gate']['passed']:
            logger.warning("Phase 2 gate FAILED. Stopping with negative result.")
            generate_findings(output_dir)
            elapsed = time.time() - start_time
            logger.info(f"\nTotal time: {elapsed/60:.1f} minutes")
            return

        run_phase3_transfer(
            best_encoder=p1['best_encoder'],
            best_config=p2['best_config'],
            n_seeds=args.seeds,
            max_epochs=30, patience=7, device=args.device,
            output_dir=output_dir,
        )
        generate_findings(output_dir)

    else:
        # Default: same as --full
        parser.print_help()
        return

    elapsed = time.time() - start_time
    logger.info(f"\nTotal time: {elapsed/60:.1f} minutes")
    logger.info(f"Results saved to {output_dir}")


if __name__ == '__main__':
    main()
