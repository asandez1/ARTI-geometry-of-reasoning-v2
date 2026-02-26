#!/usr/bin/env python3
"""
Diagnostic unit tests for the detection-application gap.

Tests every link in the chain:
  1. TypeClassifier produces different types for different inputs
  2. Different types map to different depth/delta configs
  3. Manifold with FORCED non-uniform weights produces different operators
  4. Different depths produce different tree outputs
  5. Different tree outputs produce different answer scores
  6. Fast path vs structured path produce different scores (or identical?)
  7. Controller routing actually changes depth
  8. End-to-end: inject synthetic non-uniform weights -> does accuracy change?

If any link is broken, the "zero lift" is a bug, not a finding.
"""

import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent.parent
sys.path.insert(0, str(BASE_DIR.parent))

from shared.model import COFRN, COFRNConfig
from shared.controller import (
    GeometricReasoningController, ControllerConfig,
    TypeClassifier, CORE_TYPE_NAMES, N_CORE_TYPES,
)
from shared.factorization import ContinuousFactorizationModule
from shared.operators import ContinuousOperatorManifold
from shared.reasoning_engine import ReasoningEngine, HilbertTreeFast, DirectScorer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
WARN = "\033[93mWARN\033[0m"

results = []


def report(name, passed, detail=""):
    status = PASS if passed else FAIL
    results.append((name, passed, detail))
    print(f"  [{status}] {name}")
    if detail:
        print(f"         {detail}")


def make_config():
    return COFRNConfig(
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


def load_trained_cofrn():
    """Load the actual trained model from E5."""
    config = make_config()
    model_path = BASE_DIR / "results" / "controller" / "cofrn_quick.pt"
    if not model_path.exists():
        return None, config
    cofrn = COFRN(config)
    cofrn.load_state_dict(
        torch.load(str(model_path), weights_only=True, map_location=DEVICE)
    )
    cofrn.to(DEVICE).eval()
    return cofrn, config


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 1: Manifold attention weights — are they REALLY uniform?
# ═══════════════════════════════════════════════════════════════════════════════

def test_1_attention_weights():
    print("\n=== TEST 1: Are attention weights truly uniform? ===")
    cofrn, config = load_trained_cofrn()
    if cofrn is None:
        report("T1: Load model", False, "Model not found")
        return

    torch.manual_seed(SEED)
    # Test with diverse inputs
    n_inputs = 50
    embeddings = torch.randn(n_inputs, 384, device=DEVICE)

    with torch.no_grad():
        s0 = cofrn.encode_text(embeddings=embeddings)
        fact_out = cofrn.factorization(s0)
        weights = fact_out['weights']  # [N, 16]

    # Check uniformity
    expected_uniform = 1.0 / config.n_anchors
    max_dev = (weights - expected_uniform).abs().max().item()
    mean_entropy = -(weights * torch.log(weights + 1e-10)).sum(dim=-1).mean().item()
    max_entropy = np.log(config.n_anchors)

    report("T1a: Max deviation from uniform",
           max_dev < 0.01,
           f"max_dev={max_dev:.6f} (1/16={expected_uniform:.4f})")

    report("T1b: Entropy vs max entropy",
           mean_entropy > 0.99 * max_entropy,
           f"H={mean_entropy:.4f}, H_max={max_entropy:.4f}, ratio={mean_entropy/max_entropy:.4f}")

    # Check if weights vary AT ALL across inputs
    weight_std_per_anchor = weights.std(dim=0)  # [16]
    max_input_variation = weight_std_per_anchor.max().item()
    report("T1c: Weight variation across inputs",
           max_input_variation < 1e-3,
           f"max_std_across_inputs={max_input_variation:.6f}")

    # KEY DEBUG: Look at the manifold projection
    structural = fact_out['structural']  # [N, 128]
    manifold_coords = cofrn.factorization.manifold.manifold_proj(structural)  # [N, 10]
    coord_std = manifold_coords.std(dim=0)
    coord_mean_std = coord_std.mean().item()

    report("T1d: Manifold coords vary across inputs",
           coord_mean_std > 0.01,
           f"mean_std_per_dim={coord_mean_std:.6f}")

    # Check the attention LOGITS before softmax
    anchors = cofrn.factorization.manifold.anchors  # [16, 10]
    tau = cofrn.factorization.manifold.tau
    logits = manifold_coords @ anchors.T / (tau * np.sqrt(config.manifold_dim))
    logit_range = (logits.max(dim=-1).values - logits.min(dim=-1).values).mean().item()

    report("T1e: Pre-softmax logit range",
           True,  # Informational
           f"mean_logit_range={logit_range:.4f}, tau={tau:.4f}")

    # What if we set tau very low (sharp attention)?
    sharp_logits = manifold_coords @ anchors.T / (0.01 * np.sqrt(config.manifold_dim))
    sharp_weights = F.softmax(sharp_logits, dim=-1)
    sharp_entropy = -(sharp_weights * torch.log(sharp_weights + 1e-10)).sum(dim=-1).mean().item()

    report("T1f: Entropy with tau=0.01 (forced sharp)",
           sharp_entropy < 0.5 * max_entropy,
           f"sharp_H={sharp_entropy:.4f} vs max_H={max_entropy:.4f}")

    return weights, fact_out, cofrn


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 2: Does applying operator with DIFFERENT weights produce different outputs?
# ═══════════════════════════════════════════════════════════════════════════════

def test_2_operator_sensitivity():
    print("\n=== TEST 2: Do different anchor weights produce different operators? ===")
    cofrn, config = load_trained_cofrn()
    if cofrn is None:
        report("T2: Load model", False, "Model not found")
        return

    torch.manual_seed(SEED)
    s0 = torch.randn(1, 256, device=DEVICE)

    manifold = cofrn.factorization.manifold

    # Uniform weights
    w_uniform = torch.ones(1, 16, device=DEVICE) / 16
    out_uniform = manifold.apply_operator(s0, w_uniform)

    # One-hot weights (each anchor alone)
    outputs_per_anchor = []
    for i in range(16):
        w_hot = torch.zeros(1, 16, device=DEVICE)
        w_hot[0, i] = 1.0
        out_hot = manifold.apply_operator(s0, w_hot)
        outputs_per_anchor.append(out_hot)

    # Measure pairwise distance between single-anchor outputs
    dists = []
    for i in range(16):
        for j in range(i+1, 16):
            d = (outputs_per_anchor[i] - outputs_per_anchor[j]).norm().item()
            dists.append(d)

    mean_dist = np.mean(dists)
    report("T2a: Anchors produce different operators",
           mean_dist > 0.01,
           f"mean_pairwise_L2={mean_dist:.4f}, min={min(dists):.4f}, max={max(dists):.4f}")

    # Compare uniform vs single anchors
    dists_from_uniform = []
    for i in range(16):
        d = (out_uniform - outputs_per_anchor[i]).norm().item()
        dists_from_uniform.append(d)

    report("T2b: Single anchors differ from uniform average",
           np.mean(dists_from_uniform) > 0.01,
           f"mean_dist_from_uniform={np.mean(dists_from_uniform):.4f}")

    # Compare two very different weight distributions
    w_sharp_a = torch.zeros(1, 16, device=DEVICE)
    w_sharp_a[0, 0] = 0.8
    w_sharp_a[0, 1] = 0.2
    out_a = manifold.apply_operator(s0, w_sharp_a)

    w_sharp_b = torch.zeros(1, 16, device=DEVICE)
    w_sharp_b[0, 8] = 0.8
    w_sharp_b[0, 9] = 0.2
    out_b = manifold.apply_operator(s0, w_sharp_b)

    dist_ab = (out_a - out_b).norm().item()
    report("T2c: Concentrated on different anchors -> different output",
           dist_ab > 0.01,
           f"dist(anchor[0,1] vs anchor[8,9])={dist_ab:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 3: Does tree depth actually change the output?
# ═══════════════════════════════════════════════════════════════════════════════

def test_3_depth_sensitivity():
    print("\n=== TEST 3: Does tree depth change output scores? ===")
    cofrn, config = load_trained_cofrn()
    if cofrn is None:
        report("T3: Load model", False, "Model not found")
        return

    torch.manual_seed(SEED)
    B = 4
    n_ans = 4
    embeddings = torch.randn(B, 384, device=DEVICE)
    answer_embs = torch.randn(B, n_ans, 384, device=DEVICE)

    with torch.no_grad():
        s0 = cofrn.encode_text(embeddings=embeddings)
        fact_out = cofrn.factorization(s0)
        structural = fact_out['structural']
        transformed = fact_out['transformed']
        answer_enc = cofrn.encode_answers(answer_embs)
        operator_fn = cofrn.factorization.manifold.forward

    # Run tree at different depths
    scores_by_depth = {}
    for depth in [1, 2, 3, 5]:
        with torch.no_grad():
            out = cofrn.reasoning.forward_multistep(
                transformed=transformed,
                evidence=transformed,
                operator_fn=operator_fn,
                structural=structural,
                answer_encodings=answer_enc,
                depth=depth,
            )
        scores_by_depth[depth] = out['scores'].clone()

    # Compare scores at different depths
    for d1, d2 in [(1, 2), (1, 3), (2, 3), (1, 5)]:
        diff = (scores_by_depth[d1] - scores_by_depth[d2]).abs().max().item()
        report(f"T3a: Depth {d1} vs {d2} score difference",
               diff > 1e-6,
               f"max_abs_diff={diff:.6f}")

    # Check if depth changes the PREDICTIONS
    preds_by_depth = {d: s.argmax(dim=-1) for d, s in scores_by_depth.items()}
    pred_changes = sum(
        (preds_by_depth[1] != preds_by_depth[d]).sum().item()
        for d in [2, 3, 5]
    )
    report(f"T3b: Do predictions change with depth? ({B} examples)",
           True,  # Informational
           f"total_prediction_changes_vs_depth1={pred_changes}/{B*3}")

    # KEY: Compare tree output vs direct scorer output
    with torch.no_grad():
        direct_out = cofrn.reasoning.forward_direct(
            transformed=transformed,
            answer_encodings=answer_enc,
        )
    direct_scores = direct_out['scores']

    diff_tree_vs_direct = (scores_by_depth[1] - direct_scores).abs().max().item()
    pred_diff = (scores_by_depth[1].argmax(-1) != direct_scores.argmax(-1)).sum().item()

    report("T3c: Tree(depth=1) vs DirectScorer",
           True,  # Informational
           f"max_score_diff={diff_tree_vs_direct:.6f}, pred_changes={pred_diff}/{B}")

    # Check with larger depth
    diff_tree3_vs_direct = (scores_by_depth[3] - direct_scores).abs().max().item()
    pred_diff3 = (scores_by_depth[3].argmax(-1) != direct_scores.argmax(-1)).sum().item()

    report("T3d: Tree(depth=3) vs DirectScorer",
           True,  # Informational
           f"max_score_diff={diff_tree3_vs_direct:.6f}, pred_changes={pred_diff3}/{B}")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 4: Does the tree operator_fn get called with different weights when
#          we force non-uniform attention?
# ═══════════════════════════════════════════════════════════════════════════════

def test_4_forced_nonuniform():
    print("\n=== TEST 4: Force non-uniform attention -> does accuracy change? ===")
    cofrn, config = load_trained_cofrn()
    if cofrn is None:
        report("T4: Load model", False, "Model not found")
        return

    # Load real benchmark data
    try:
        from shared.data_utils import precompute_embeddings_st
        ds = precompute_embeddings_st('all-MiniLM-L6-v2', 'gsm8k', 'test', seed=SEED)
    except Exception as e:
        report("T4: Load data", False, f"Error: {e}")
        return

    rng = np.random.RandomState(SEED)
    n = 100
    indices = rng.permutation(len(ds))[:n]
    q_emb = ds.question_embeddings[indices].to(DEVICE)
    a_emb = ds.answer_embeddings[indices].to(DEVICE)
    labels = ds.labels[indices].to(DEVICE)

    # Baseline: normal forward (uniform attention)
    with torch.no_grad():
        out_normal = cofrn(embeddings=q_emb, answer_embeddings=a_emb, labels=labels)
    normal_scores = out_normal['scores']
    normal_preds = normal_scores.argmax(-1)
    normal_acc = (normal_preds == labels).float().mean().item()

    # Monkeypatch: force anchor 0 to get all weight
    original_compute = cofrn.factorization.manifold.compute_attention_weights

    def forced_anchor0(structural):
        w = torch.zeros(structural.shape[0], 16, device=structural.device)
        w[:, 0] = 1.0
        return w

    def forced_anchor12(structural):
        w = torch.zeros(structural.shape[0], 16, device=structural.device)
        w[:, 12] = 1.0
        return w

    def forced_sharp_varied(structural):
        """Use the actual logits but with very low temperature."""
        manifold = cofrn.factorization.manifold
        coords = manifold.manifold_proj(structural)
        logits = coords @ manifold.anchors.T / (0.01 * np.sqrt(config.manifold_dim))
        return F.softmax(logits, dim=-1)

    # Test with forced anchor 0
    cofrn.factorization.manifold.compute_attention_weights = forced_anchor0
    with torch.no_grad():
        out_a0 = cofrn(embeddings=q_emb, answer_embeddings=a_emb, labels=labels)
    a0_preds = out_a0['scores'].argmax(-1)
    a0_acc = (a0_preds == labels).float().mean().item()

    # Test with forced anchor 12
    cofrn.factorization.manifold.compute_attention_weights = forced_anchor12
    with torch.no_grad():
        out_a12 = cofrn(embeddings=q_emb, answer_embeddings=a_emb, labels=labels)
    a12_preds = out_a12['scores'].argmax(-1)
    a12_acc = (a12_preds == labels).float().mean().item()

    # Test with forced sharp (actual input-dependent but sharp)
    cofrn.factorization.manifold.compute_attention_weights = forced_sharp_varied
    with torch.no_grad():
        out_sharp = cofrn(embeddings=q_emb, answer_embeddings=a_emb, labels=labels)
    sharp_preds = out_sharp['scores'].argmax(-1)
    sharp_acc = (sharp_preds == labels).float().mean().item()

    # Restore
    cofrn.factorization.manifold.compute_attention_weights = original_compute

    report("T4a: Normal (uniform) accuracy",
           True,
           f"acc={normal_acc:.1%}")

    report("T4b: Forced anchor 0 accuracy",
           True,
           f"acc={a0_acc:.1%} (Δ={a0_acc-normal_acc:+.1%})")

    report("T4c: Forced anchor 12 accuracy",
           True,
           f"acc={a12_acc:.1%} (Δ={a12_acc-normal_acc:+.1%})")

    report("T4d: Forced sharp (tau=0.01) accuracy",
           True,
           f"acc={sharp_acc:.1%} (Δ={sharp_acc-normal_acc:+.1%})")

    pred_changes_a0 = (normal_preds != a0_preds).sum().item()
    pred_changes_a12 = (normal_preds != a12_preds).sum().item()
    pred_changes_sharp = (normal_preds != sharp_preds).sum().item()
    pred_changes_a0_vs_a12 = (a0_preds != a12_preds).sum().item()

    report("T4e: Prediction changes (uniform vs anchor0)",
           pred_changes_a0 > 0,
           f"changes={pred_changes_a0}/{n}")

    report("T4f: Prediction changes (uniform vs anchor12)",
           pred_changes_a12 > 0,
           f"changes={pred_changes_a12}/{n}")

    report("T4g: Prediction changes (anchor0 vs anchor12)",
           pred_changes_a0_vs_a12 > 0,
           f"changes={pred_changes_a0_vs_a12}/{n}")

    report("T4h: Prediction changes (uniform vs sharp)",
           pred_changes_sharp > 0,
           f"changes={pred_changes_sharp}/{n}")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 5: Controller routing — does it actually change depth?
# ═══════════════════════════════════════════════════════════════════════════════

def test_5_controller_routing():
    print("\n=== TEST 5: Controller routing changes depth for different types ===")
    cofrn, config = load_trained_cofrn()
    if cofrn is None:
        report("T5: Load model", False, "Model not found")
        return

    clf_path = BASE_DIR / "results" / "controller" / "type_clf.pt"
    if not clf_path.exists():
        report("T5: Load classifier", False, "type_clf.pt not found")
        return

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

    # Create synthetic inputs that should map to different types
    torch.manual_seed(SEED)
    n = 200
    s0 = torch.randn(n, 256, device=DEVICE)

    with torch.no_grad():
        fact_out = cofrn.factorization(s0)
        structural = fact_out['structural']
        tp, dt, conf, mc = controller.detect_type(s0, structural)

    type_counts = defaultdict(int)
    for t in dt.cpu().tolist():
        type_counts[CORE_TYPE_NAMES[t]] += 1

    report("T5a: TypeClassifier produces diverse types",
           len(type_counts) >= 3,
           f"distribution={dict(type_counts)}")

    # Verify different types -> different configs
    configs_seen = set()
    for t in range(N_CORE_TYPES):
        depth, delta = controller.get_type_config(t)
        configs_seen.add((depth, delta))

    report("T5b: Different types -> different depth/delta",
           len(configs_seen) >= 3,
           f"unique_configs={len(configs_seen)}: {configs_seen}")

    # Verify routing weights are not all identical
    with torch.no_grad():
        rw = controller.compute_routing(tp, conf, fact_out['weights'])
    rw_std = rw.std().item()
    rw_mean = rw.mean().item()

    report("T5c: Routing weights vary across inputs",
           rw_std > 0.01,
           f"mean={rw_mean:.4f}, std={rw_std:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 6: End-to-end controller — does it ACTUALLY run different depths?
# ═══════════════════════════════════════════════════════════════════════════════

def test_6_e2e_controller_depth():
    print("\n=== TEST 6: End-to-end — does the controller's depth routing work? ===")
    cofrn, config = load_trained_cofrn()
    if cofrn is None:
        report("T6: Load model", False, "Model not found")
        return

    torch.manual_seed(SEED)
    B = 8
    n_ans = 4
    embeddings = torch.randn(B, 384, device=DEVICE)
    answer_embs = torch.randn(B, n_ans, 384, device=DEVICE)

    with torch.no_grad():
        s0 = cofrn.encode_text(embeddings=embeddings)
        fact_out = cofrn.factorization(s0)
        structural = fact_out['structural']
        transformed = fact_out['transformed']
        answer_enc = cofrn.encode_answers(answer_embs)
        operator_fn = cofrn.factorization.manifold.forward

    # Run same input at depth=1 and depth=3 manually
    depths = [1, 2, 3]
    deltas = [0.9, 0.6, 0.7]
    all_scores = {}

    for depth, delta in zip(depths, deltas):
        old_delta = cofrn.reasoning.tree.delta
        cofrn.reasoning.tree.delta = delta
        with torch.no_grad():
            out = cofrn.reasoning.forward_multistep(
                transformed=transformed,
                evidence=transformed,
                operator_fn=operator_fn,
                structural=structural,
                answer_encodings=answer_enc,
                depth=depth,
            )
        cofrn.reasoning.tree.delta = old_delta
        all_scores[(depth, delta)] = out['scores'].clone()

    # Compare
    for i, (d1, del1) in enumerate(zip(depths, deltas)):
        for d2, del2 in zip(depths[i+1:], deltas[i+1:]):
            diff = (all_scores[(d1, del1)] - all_scores[(d2, del2)]).abs().max().item()
            pred_diff = (all_scores[(d1, del1)].argmax(-1) != all_scores[(d2, del2)].argmax(-1)).sum().item()
            report(f"T6a: depth={d1}/δ={del1} vs depth={d2}/δ={del2}",
                   True,  # informational
                   f"max_score_diff={diff:.6f}, pred_changes={pred_diff}/{B}")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 7: The SCORING function — are scores dominated by answer encodings?
# ═══════════════════════════════════════════════════════════════════════════════

def test_7_scoring_sensitivity():
    print("\n=== TEST 7: How sensitive are answer scores to transformed state? ===")
    cofrn, config = load_trained_cofrn()
    if cofrn is None:
        report("T7: Load model", False, "Model not found")
        return

    torch.manual_seed(SEED)
    B = 4
    n_ans = 4
    answer_embs = torch.randn(B, n_ans, 384, device=DEVICE)

    with torch.no_grad():
        answer_enc = cofrn.encode_answers(answer_embs)

    # Create two very different transformed states
    t1 = torch.randn(B, 256, device=DEVICE)
    t2 = -t1  # Maximally different

    with torch.no_grad():
        out1 = cofrn.reasoning.forward_direct(transformed=t1, answer_encodings=answer_enc)
        out2 = cofrn.reasoning.forward_direct(transformed=t2, answer_encodings=answer_enc)

    score_diff = (out1['scores'] - out2['scores']).abs().max().item()
    pred_diff = (out1['scores'].argmax(-1) != out2['scores'].argmax(-1)).sum().item()

    report("T7a: DirectScorer sensitive to transformed state",
           score_diff > 0.1,
           f"max_score_diff={score_diff:.4f}, pred_changes={pred_diff}/{B}")

    # Small perturbation
    t3 = t1 + 0.01 * torch.randn_like(t1)
    with torch.no_grad():
        out3 = cofrn.reasoning.forward_direct(transformed=t3, answer_encodings=answer_enc)

    small_diff = (out1['scores'] - out3['scores']).abs().max().item()
    report("T7b: DirectScorer sensitivity to small perturbation",
           small_diff > 0,
           f"max_score_diff for 0.01*noise: {small_diff:.6f}")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 8: The REAL test — per-example depth routing on actual data
# ═══════════════════════════════════════════════════════════════════════════════

def test_8_real_benchmark_depth_sweep():
    print("\n=== TEST 8: Depth sweep on real GSM8K data ===")
    cofrn, config = load_trained_cofrn()
    if cofrn is None:
        report("T8: Load model", False, "Model not found")
        return

    try:
        from shared.data_utils import precompute_embeddings_st
        ds = precompute_embeddings_st('all-MiniLM-L6-v2', 'gsm8k', 'test', seed=SEED)
    except Exception as e:
        report("T8: Load data", False, f"Error: {e}")
        return

    rng = np.random.RandomState(SEED)
    n = 100
    indices = rng.permutation(len(ds))[:n]
    q_emb = ds.question_embeddings[indices].to(DEVICE)
    a_emb = ds.answer_embeddings[indices].to(DEVICE)
    labels = ds.labels[indices].to(DEVICE)

    # Run at each depth
    accs = {}
    all_preds = {}
    for depth in [1, 2, 3, 5]:
        correct = 0
        preds_list = []
        with torch.no_grad():
            for start in range(0, n, 32):
                end = min(start + 32, n)
                s0 = cofrn.encode_text(embeddings=q_emb[start:end])
                fact_out = cofrn.factorization(s0)
                out = cofrn.reasoning.forward_multistep(
                    transformed=fact_out['transformed'],
                    evidence=fact_out['transformed'],
                    operator_fn=cofrn.factorization.manifold.forward,
                    structural=fact_out['structural'],
                    answer_encodings=cofrn.encode_answers(a_emb[start:end]),
                    depth=depth,
                )
                pred = out['scores'].argmax(-1)
                correct += (pred == labels[start:end]).sum().item()
                preds_list.append(pred.cpu())

        accs[depth] = correct / n
        all_preds[depth] = torch.cat(preds_list)
        report(f"T8a: GSM8K accuracy at depth={depth}",
               True,
               f"acc={accs[depth]:.1%}")

    # Direct scorer comparison
    direct_correct = 0
    direct_preds = []
    with torch.no_grad():
        for start in range(0, n, 32):
            end = min(start + 32, n)
            s0 = cofrn.encode_text(embeddings=q_emb[start:end])
            fact_out = cofrn.factorization(s0)
            out = cofrn.reasoning.forward_direct(
                transformed=fact_out['transformed'],
                answer_encodings=cofrn.encode_answers(a_emb[start:end]),
            )
            pred = out['scores'].argmax(-1)
            direct_correct += (pred == labels[start:end]).sum().item()
            direct_preds.append(pred.cpu())

    direct_acc = direct_correct / n
    direct_preds = torch.cat(direct_preds)

    report(f"T8b: GSM8K DirectScorer accuracy",
           True,
           f"acc={direct_acc:.1%}")

    # Prediction agreement
    for depth in [1, 2, 3, 5]:
        agree = (all_preds[depth] == direct_preds).sum().item()
        report(f"T8c: Tree(depth={depth}) vs Direct: prediction agreement",
               True,
               f"agree={agree}/{n} ({agree/n:.1%})")

    # KEY: Are different depths producing the SAME predictions?
    all_same = True
    for d in [2, 3, 5]:
        if not torch.equal(all_preds[1], all_preds[d]):
            all_same = False
            break

    report("T8d: All depths produce IDENTICAL predictions?",
           not all_same,  # PASS if they differ
           f"identical={'YES — THIS IS THE BUG' if all_same else 'No, they differ'}")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 9: Trace through the tree to see what happens step by step
# ═══════════════════════════════════════════════════════════════════════════════

def test_9_tree_trace():
    print("\n=== TEST 9: Step-by-step tree trace ===")
    cofrn, config = load_trained_cofrn()
    if cofrn is None:
        report("T9: Load model", False, "Model not found")
        return

    torch.manual_seed(SEED)
    embeddings = torch.randn(1, 384, device=DEVICE)

    with torch.no_grad():
        s0 = cofrn.encode_text(embeddings=embeddings)
        fact_out = cofrn.factorization(s0)
        transformed = fact_out['transformed']
        structural = fact_out['structural']

    tree = cofrn.reasoning.tree
    manifold = cofrn.factorization.manifold

    # Manual tree stepping
    print("    Step-by-step tree evolution:")
    state = transformed.unsqueeze(1).expand(-1, tree.beam_width, -1).clone()
    # Add diversity
    diversity = tree.diversity_vectors[:tree.beam_width].unsqueeze(0)
    state = F.normalize(state + 0.01 * diversity, dim=-1)

    for depth in range(5):
        B, beam, D = state.shape
        flat_state = state.reshape(B * beam, D)
        flat_struct = structural.unsqueeze(1).expand(-1, beam, -1).reshape(B * beam, -1)

        # Apply operator
        new_state, weights = manifold(flat_state, flat_struct)
        new_state = new_state.reshape(B, beam, D)

        # Measure change
        change = (new_state - state).norm(dim=-1).mean().item()
        cosine = F.cosine_similarity(
            state.reshape(-1, D), new_state.reshape(-1, D)
        ).mean().item()

        w = weights.reshape(B, beam, -1)
        w_entropy = -(w * torch.log(w + 1e-10)).sum(dim=-1).mean().item()

        print(f"    Depth {depth}: Δnorm={change:.4f}, cos_sim={cosine:.4f}, "
              f"w_entropy={w_entropy:.4f}")

        state = new_state

    report("T9: Tree trace complete", True, "See step-by-step output above")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  PIPELINE DEBUG: Detection-Application Gap Diagnosis")
    print("=" * 70)

    test_1_attention_weights()
    test_2_operator_sensitivity()
    test_3_depth_sensitivity()
    test_4_forced_nonuniform()
    test_5_controller_routing()
    test_6_e2e_controller_depth()
    test_7_scoring_sensitivity()
    test_8_real_benchmark_depth_sweep()
    test_9_tree_trace()

    # Summary
    n_pass = sum(1 for _, p, _ in results if p)
    n_fail = sum(1 for _, p, _ in results if not p)
    print(f"\n{'='*70}")
    print(f"  SUMMARY: {n_pass} passed, {n_fail} failed out of {len(results)} tests")
    print(f"{'='*70}")

    if n_fail > 0:
        print("\n  FAILURES:")
        for name, passed, detail in results:
            if not passed:
                print(f"    [{FAIL}] {name}: {detail}")

    # Key diagnostic summary
    print(f"\n  KEY DIAGNOSTIC QUESTIONS:")
    print(f"  Q1: Are attention weights uniform? -> Check T1a-T1c")
    print(f"  Q2: Can different weights produce different operators? -> Check T2a-T2c")
    print(f"  Q3: Does depth change output? -> Check T3a-T3d")
    print(f"  Q4: Does forcing non-uniform weights change accuracy? -> Check T4a-T4h")
    print(f"  Q5: Does the controller route to different depths? -> Check T5a-T5c")
    print(f"  Q6: On real data, do different depths give different accuracy? -> Check T8a-T8d")


if __name__ == "__main__":
    main()
