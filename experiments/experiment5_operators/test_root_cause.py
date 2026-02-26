#!/usr/bin/env python3
"""
Root cause diagnosis for the zero-lift problem.

Tests 3 specific hypotheses:
  H1: Operators are near-identity (U,V barely moved from 0.01 init)
  H2: tree_scorer ignores best_hyp (learned to depend only on answer_enc)
  H3: Operator perturbation is too small to survive L2 normalization

If H2 is true -> tree_scorer is architecturally decoupled from the manifold
If H2 is false but H1+H3 true -> operators need larger scale / different training
"""

import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR.parent))

from shared.model import COFRN, COFRNConfig
from shared.data_utils import precompute_embeddings_st

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


def load_model():
    config = COFRNConfig(
        use_precomputed=True, encoder_input_dim=384, hidden_dim=256,
        struct_dim=128, context_dim=128, manifold_dim=10,
        n_anchors=16, rank=16, task_type='multi_step',
    )
    model_path = BASE_DIR / "results" / "controller" / "cofrn_quick.pt"
    cofrn = COFRN(config)
    cofrn.load_state_dict(torch.load(str(model_path), weights_only=True, map_location=DEVICE))
    cofrn.to(DEVICE).eval()
    return cofrn, config


def load_data(n=100):
    ds = precompute_embeddings_st('all-MiniLM-L6-v2', 'gsm8k', 'test', seed=SEED)
    rng = np.random.RandomState(SEED)
    indices = rng.permutation(len(ds))[:n]
    q = ds.question_embeddings[indices].to(DEVICE)
    a = ds.answer_embeddings[indices].to(DEVICE)
    l = ds.labels[indices].to(DEVICE)
    return q, a, l


def predict_tree(cofrn, q, a, depth=3):
    """Run tree path, return predictions and best_hyp."""
    with torch.no_grad():
        s0 = cofrn.encode_text(embeddings=q)
        fact = cofrn.factorization(s0)
        answer_enc = cofrn.encode_answers(a)
        tree_out = cofrn.reasoning.tree(
            initial_state=fact['transformed'],
            evidence=fact['transformed'],
            operator_fn=cofrn.factorization.manifold.forward,
            structural=fact['structural'],
            depth=depth,
        )
        h_w = F.softmax(tree_out['hypothesis_scores'], dim=-1)
        best_hyp = (h_w.unsqueeze(-1) * tree_out['hypothesis_states']).sum(dim=1)

        best_exp = best_hyp.unsqueeze(1).expand_as(answer_enc)
        paired = torch.cat([best_exp, answer_enc], dim=-1)
        scores = cofrn.reasoning.tree_scorer(paired).squeeze(-1)
        preds = scores.argmax(-1)
    return preds, scores, best_hyp, fact['transformed']


def main():
    print("=" * 70)
    print("  ROOT CAUSE DIAGNOSIS: Why Zero Lift?")
    print("=" * 70)

    cofrn, config = load_model()
    q, a, labels = load_data(100)

    # ═══════════════════════════════════════════════════════════════════
    # H1: How far did operators move from initialization?
    # ═══════════════════════════════════════════════════════════════════
    print("\n--- H1: Operator Scale ---")
    U = cofrn.factorization.manifold.U_all  # [16, 256, 16]
    V = cofrn.factorization.manifold.V_all  # [16, 256, 16]

    u_rms = U.detach().pow(2).mean().sqrt().item()
    v_rms = V.detach().pow(2).mean().sqrt().item()
    u_max = U.detach().abs().max().item()
    v_max = V.detach().abs().max().item()

    # Expected init scale: randn * 0.01 -> std ≈ 0.01
    print(f"  U_all RMS: {u_rms:.6f} (init ≈ 0.01)")
    print(f"  V_all RMS: {v_rms:.6f} (init ≈ 0.01)")
    print(f"  U_all max: {u_max:.6f}")
    print(f"  V_all max: {v_max:.6f}")
    print(f"  Growth from init: U={u_rms/0.01:.2f}x, V={v_rms/0.01:.2f}x")

    # Measure actual perturbation magnitude on real data
    with torch.no_grad():
        s0 = cofrn.encode_text(embeddings=q[:32])
        fact = cofrn.factorization(s0)
        transformed = fact['transformed']

    perturbation = (transformed - s0[:32]).norm(dim=-1)
    # But s0 and transformed may have different scales due to encoder projection
    # Let's measure the operator perturbation directly
    with torch.no_grad():
        x = fact['transformed'][:32]
        w = fact['weights'][:32]
        manifold = cofrn.factorization.manifold

        Vx = torch.einsum('ahr,bh->bar', manifold.V_all, x)
        UVx = torch.einsum('ahr,bar->bah', manifold.U_all, Vx)
        weighted_pert = torch.einsum('ba,bah->bh', w, UVx)
        pert_norm = weighted_pert.norm(dim=-1)
        x_norm = x.norm(dim=-1)
        relative_pert = (pert_norm / x_norm).mean().item()

    print(f"  Actual perturbation |UVx|/|x|: {relative_pert:.6f}")
    print(f"  Actual perturbation |UVx| mean: {pert_norm.mean().item():.6f}")

    # ═══════════════════════════════════════════════════════════════════
    # H2: Does tree_scorer use best_hyp at all?
    # ═══════════════════════════════════════════════════════════════════
    print("\n--- H2: Does tree_scorer use best_hyp? ---")

    preds_normal, scores_normal, best_hyp, transformed = predict_tree(cofrn, q, a)
    normal_acc = (preds_normal == labels).float().mean().item()
    print(f"  Normal tree accuracy: {normal_acc:.1%}")

    # Test 2a: Replace best_hyp with RANDOM vectors
    with torch.no_grad():
        answer_enc = cofrn.encode_answers(a)
        torch.manual_seed(999)
        random_hyp = F.normalize(torch.randn_like(best_hyp), dim=-1)
        rand_exp = random_hyp.unsqueeze(1).expand_as(answer_enc)
        paired_rand = torch.cat([rand_exp, answer_enc], dim=-1)
        scores_rand = cofrn.reasoning.tree_scorer(paired_rand).squeeze(-1)
        preds_rand = scores_rand.argmax(-1)

    rand_acc = (preds_rand == labels).float().mean().item()
    changes_rand = (preds_normal != preds_rand).sum().item()
    print(f"  Random best_hyp accuracy: {rand_acc:.1%}")
    print(f"  Pred changes (normal vs random hyp): {changes_rand}/100")

    # Test 2b: Replace best_hyp with ZEROS
    with torch.no_grad():
        zero_hyp = torch.zeros_like(best_hyp)
        zero_exp = zero_hyp.unsqueeze(1).expand_as(answer_enc)
        paired_zero = torch.cat([zero_exp, answer_enc], dim=-1)
        scores_zero = cofrn.reasoning.tree_scorer(paired_zero).squeeze(-1)
        preds_zero = scores_zero.argmax(-1)

    zero_acc = (preds_zero == labels).float().mean().item()
    changes_zero = (preds_normal != preds_zero).sum().item()
    print(f"  Zero best_hyp accuracy: {zero_acc:.1%}")
    print(f"  Pred changes (normal vs zero hyp): {changes_zero}/100")

    # Test 2c: Replace best_hyp with NEGATED version
    with torch.no_grad():
        neg_hyp = -best_hyp
        neg_exp = neg_hyp.unsqueeze(1).expand_as(answer_enc)
        paired_neg = torch.cat([neg_exp, answer_enc], dim=-1)
        scores_neg = cofrn.reasoning.tree_scorer(paired_neg).squeeze(-1)
        preds_neg = scores_neg.argmax(-1)

    neg_acc = (preds_neg == labels).float().mean().item()
    changes_neg = (preds_normal != preds_neg).sum().item()
    print(f"  Negated best_hyp accuracy: {neg_acc:.1%}")
    print(f"  Pred changes (normal vs negated hyp): {changes_neg}/100")

    # Test 2d: Replace best_hyp with ANOTHER question's best_hyp (shuffled)
    with torch.no_grad():
        perm = torch.randperm(100, device=DEVICE)
        shuffled_hyp = best_hyp[perm]
        shuf_exp = shuffled_hyp.unsqueeze(1).expand_as(answer_enc)
        paired_shuf = torch.cat([shuf_exp, answer_enc], dim=-1)
        scores_shuf = cofrn.reasoning.tree_scorer(paired_shuf).squeeze(-1)
        preds_shuf = scores_shuf.argmax(-1)

    shuf_acc = (preds_shuf == labels).float().mean().item()
    changes_shuf = (preds_normal != preds_shuf).sum().item()
    print(f"  Shuffled best_hyp accuracy: {shuf_acc:.1%}")
    print(f"  Pred changes (normal vs shuffled hyp): {changes_shuf}/100")

    # Verdict for H2
    uses_hyp = changes_rand > 10 or changes_zero > 10 or changes_neg > 10
    print(f"\n  >>> H2 VERDICT: tree_scorer {'USES' if uses_hyp else 'IGNORES'} best_hyp")
    if not uses_hyp:
        print("  >>> The MLP has learned to score based ONLY on answer_enc structure")
        print("  >>> This means no upstream change (operators, anchors, depth) can affect predictions")

    # ═══════════════════════════════════════════════════════════════════
    # H3: If we scale up operators, do predictions change?
    # ═══════════════════════════════════════════════════════════════════
    print("\n--- H3: Force large operators -> do predictions change? ---")

    for scale_factor in [10, 100, 1000]:
        with torch.no_grad():
            # Temporarily scale up U, V
            orig_U = cofrn.factorization.manifold.U_all.data.clone()
            orig_V = cofrn.factorization.manifold.V_all.data.clone()

            cofrn.factorization.manifold.U_all.data *= scale_factor
            cofrn.factorization.manifold.V_all.data *= scale_factor

            preds_scaled, scores_scaled, best_hyp_scaled, _ = predict_tree(cofrn, q, a)

            # Restore
            cofrn.factorization.manifold.U_all.data = orig_U
            cofrn.factorization.manifold.V_all.data = orig_V

        scaled_acc = (preds_scaled == labels).float().mean().item()
        changes = (preds_normal != preds_scaled).sum().item()
        hyp_diff = (best_hyp - best_hyp_scaled).norm(dim=-1).mean().item()

        print(f"  Scale {scale_factor}x: acc={scaled_acc:.1%}, "
              f"pred_changes={changes}/100, hyp_L2_diff={hyp_diff:.4f}")

    # ═══════════════════════════════════════════════════════════════════
    # H4: What does the tree_scorer weight matrix look like?
    # ═══════════════════════════════════════════════════════════════════
    print("\n--- H4: tree_scorer weight analysis ---")
    scorer = cofrn.reasoning.tree_scorer
    # tree_scorer is Sequential(Linear(512, 256), GELU, Linear(256, 1))
    W1 = scorer[0].weight.detach()  # [256, 512]
    b1 = scorer[0].bias.detach()    # [256]

    # Split W1 into hyp-attending and answer-attending parts
    W1_hyp = W1[:, :256]    # First 256 cols attend to best_hyp
    W1_ans = W1[:, 256:]    # Last 256 cols attend to answer_enc

    hyp_norm = W1_hyp.norm(dim=1).mean().item()
    ans_norm = W1_ans.norm(dim=1).mean().item()
    ratio = hyp_norm / (ans_norm + 1e-10)

    print(f"  W1 hyp-half norm: {hyp_norm:.4f}")
    print(f"  W1 ans-half norm: {ans_norm:.4f}")
    print(f"  Ratio (hyp/ans): {ratio:.4f}")
    print(f"  Bias norm: {b1.norm().item():.4f}")

    # Frobenius norm
    print(f"  ||W1_hyp||_F = {W1_hyp.norm().item():.4f}")
    print(f"  ||W1_ans||_F = {W1_ans.norm().item():.4f}")

    # Check if hyp-half weights are near zero
    hyp_max = W1_hyp.abs().max().item()
    ans_max = W1_ans.abs().max().item()
    print(f"  W1_hyp max: {hyp_max:.4f}")
    print(f"  W1_ans max: {ans_max:.4f}")

    # ═══════════════════════════════════════════════════════════════════
    # H5: Direct scorer weight analysis (for comparison)
    # ═══════════════════════════════════════════════════════════════════
    print("\n--- H5: direct scorer weight analysis ---")
    d_scorer = cofrn.reasoning.direct.scorer
    DW1 = d_scorer[0].weight.detach()  # [256, 512]

    DW1_trans = DW1[:, :256]
    DW1_ans = DW1[:, 256:]

    print(f"  W1 transformed-half norm: {DW1_trans.norm(dim=1).mean().item():.4f}")
    print(f"  W1 answer-half norm: {DW1_ans.norm(dim=1).mean().item():.4f}")
    print(f"  Ratio (trans/ans): {DW1_trans.norm(dim=1).mean().item() / (DW1_ans.norm(dim=1).mean().item() + 1e-10):.4f}")

    # ═══════════════════════════════════════════════════════════════════
    # H6: Is best_hyp actually near-constant across inputs?
    # ═══════════════════════════════════════════════════════════════════
    print("\n--- H6: best_hyp variation across inputs ---")

    hyp_centroid = best_hyp.mean(dim=0)  # [256]
    hyp_dists = (best_hyp - hyp_centroid.unsqueeze(0)).norm(dim=-1)
    hyp_cosines = F.cosine_similarity(
        best_hyp, hyp_centroid.unsqueeze(0).expand_as(best_hyp), dim=-1
    )

    print(f"  best_hyp L2 to centroid: mean={hyp_dists.mean().item():.4f}, "
          f"std={hyp_dists.std().item():.4f}")
    print(f"  best_hyp cosine to centroid: mean={hyp_cosines.mean().item():.4f}, "
          f"min={hyp_cosines.min().item():.4f}")

    trans_centroid = transformed.mean(dim=0)
    trans_cosines = F.cosine_similarity(
        transformed, trans_centroid.unsqueeze(0).expand_as(transformed), dim=-1
    )
    print(f"  transformed cosine to centroid: mean={trans_cosines.mean().item():.4f}, "
          f"min={trans_cosines.min().item():.4f}")

    # How similar are best_hyp and transformed?
    hyp_vs_trans = F.cosine_similarity(best_hyp, transformed, dim=-1)
    print(f"  best_hyp vs transformed cosine: mean={hyp_vs_trans.mean().item():.4f}, "
          f"min={hyp_vs_trans.min().item():.4f}")

    # ═══════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  DIAGNOSIS SUMMARY")
    print("=" * 70)
    print(f"""
  H1 (operators near-identity): U RMS={u_rms:.4f}, perturbation={relative_pert:.6f}
     -> {'CONFIRMED' if relative_pert < 0.01 else 'REJECTED'}: operators {'barely' if relative_pert < 0.01 else 'significantly'} perturb input

  H2 (tree_scorer ignores best_hyp):
     Changes with random hyp:  {changes_rand}/100
     Changes with zero hyp:    {changes_zero}/100
     Changes with negated hyp: {changes_neg}/100
     Changes with shuffled hyp:{changes_shuf}/100
     -> {'CONFIRMED' if not uses_hyp else 'REJECTED'}: tree_scorer {'ignores' if not uses_hyp else 'uses'} best_hyp

  H3 (scaling operators):
     -> Check above for scale 10x/100x/1000x results

  H4 (weight analysis):
     hyp/ans weight ratio: {ratio:.4f}
     -> {'tree_scorer attends mainly to ANSWERS' if ratio < 0.5 else 'tree_scorer uses BOTH inputs'}

  ROOT CAUSE: {'tree_scorer has learned to score based ONLY on answer encodings.' if not uses_hyp else 'Operators are too small to produce meaningful state changes.'}
  {'The manifold/operator/depth pipeline is completely disconnected from predictions.' if not uses_hyp else 'Need larger operator scale or different training to create meaningful perturbations.'}
  {'No upstream fix (rules, fine-tuning, anchor weights) can help because tree_scorer ignores the tree output.' if not uses_hyp else 'Fix: increase operator scale, retrain with auxiliary losses that force operator use.'}
""")


if __name__ == "__main__":
    main()
