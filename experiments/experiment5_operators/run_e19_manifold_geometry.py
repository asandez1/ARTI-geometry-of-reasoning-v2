"""
E19: Manifold Geometry of Reasoning Types — GCMC-Inspired Analysis.

Computes covariance-based approximations of the 6 GCMC effective geometric
measures (Chou et al. 2024, [26]) on the 10 reasoning type manifolds, then
tests whether the same geometric properties that predict task-efficiency in
biological brains also predict ARTI classification accuracy.

This is a pure-analysis experiment — no training, no GPU required.
Geometric computation on existing ARTI v3 embeddings (7,500 x 384D).

Six GCMC-inspired measures (per-type):
    R_eff     — effective radius: sqrt(trace(Sigma_k) / D)
    D_eff     — effective dimension: (sum lam)^2 / sum(lam^2)
    Center    — center norm: ||mu_k||

Six GCMC-inspired measures (pairwise, 45 pairs):
    Axes alignment     — mean |cos(PC_i, PC_j)| for top-10 PCs
    Center alignment   — cos(mu_i, mu_j) between centroids
    Center-axes align  — mean |cos(mu_i, PC_j)| cross-term

Derived per-type:
    Inter-type separation  — mean centroid distance to all others
    Compactness ratio      — R_eff / inter-type separation
    Mean axes alignment    — mean axes alignment to all others

Correlations with ARTI ensemble accuracy (N=10 types):
    Spearman + Pearson for each per-type measure vs accuracy
    OLS regression: R_eff + D_eff + compactness -> accuracy

Confusion analysis (SC4):
    Correlate off-diagonal confusion rates with pairwise center alignment

Bootstrap (B=1000): 95% CI on each geometric measure.

Success Criteria:
    SC1: Spearman rho(R_eff, accuracy) < -0.5, p < 0.10
    SC2: Spearman rho(D_eff, accuracy) < -0.4, p < 0.15
    SC3: Spearman rho(compactness, accuracy) < -0.5, p < 0.10
    SC4: Spearman rho(center_align, confusion) > 0.3, p < 0.10
    SC5: OLS R^2 > 0.60 from 3-measure regression

Usage:
    cd paper13_The_geometry_of_Machine_Reasoning
    python experiments/experiment5_operators/run_e19_manifold_geometry.py
"""

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import confusion_matrix

import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
SHARED_DIR = BASE_DIR.parent / "shared"
RESULTS_DIR = BASE_DIR / "results" / "e19_manifold_geometry"

DATASET_PATH = BASE_DIR / "results" / "arti_v3" / "dataset.pt"
ARTI_MODEL_PATH = BASE_DIR / "results" / "arti_v3" / "arti_model.pt"
ENSEMBLE_RESULTS_PATH = BASE_DIR / "results" / "arti_ensemble" / "ensemble_results.json"

sys.path.insert(0, str(SHARED_DIR.parent))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_TYPES = 10
EMBEDDING_DIM = 384
TOP_K_PCS = 10
N_BOOTSTRAP = 1000
RANDOM_SEED = 42

TYPE_NAMES = [
    "PhysCause", "BehvCause", "SysCause", "Deduc", "Induc",
    "Analog", "Conserv", "Counter", "Abduc", "Decomp",
]

TYPE_FULL_NAMES = [
    "Physical Cause", "Behavioral Cause", "Systemic Cause",
    "Deduction", "Induction", "Analogy", "Conservation",
    "Counterfactual", "Abduction", "Decomposition",
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


# ===================================================================
# Geometric Measure Functions
# ===================================================================

def compute_per_type_measures(X: np.ndarray, y: np.ndarray):
    """Compute per-type GCMC geometric measures.

    Returns dict mapping type_id -> {R_eff, D_eff, center_norm, eigvecs, mu, cov}.
    """
    measures = {}
    for k in range(N_TYPES):
        mask = y == k
        X_k = X[mask]
        n_k = X_k.shape[0]

        # Centroid
        mu_k = X_k.mean(axis=0)
        center_norm = np.linalg.norm(mu_k)

        # Covariance
        cov_k = np.cov(X_k, rowvar=False)  # [D, D]

        # Eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(cov_k)
        # Sort descending
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # Clip tiny negative eigenvalues from numerical noise
        eigvals = np.maximum(eigvals, 0.0)

        # R_eff = sqrt(trace(Sigma) / D)  — GCMC Eq. 11 approx
        trace_cov = eigvals.sum()
        R_eff = np.sqrt(trace_cov / EMBEDDING_DIM)

        # D_eff = (sum lam)^2 / sum(lam^2) — participation ratio, GCMC Eq. 10 approx
        sum_lam = eigvals.sum()
        sum_lam2 = (eigvals ** 2).sum()
        D_eff = (sum_lam ** 2) / sum_lam2 if sum_lam2 > 0 else 0.0

        measures[k] = {
            'R_eff': float(R_eff),
            'D_eff': float(D_eff),
            'center_norm': float(center_norm),
            'n_samples': int(n_k),
            'trace_cov': float(trace_cov),
            'top_eigvals': eigvals[:TOP_K_PCS].tolist(),
            'eigvecs_top': eigvecs[:, :TOP_K_PCS],  # [D, K] — not serialized
            'mu': mu_k,  # not serialized
        }
    return measures


def compute_pairwise_measures(measures: dict):
    """Compute pairwise GCMC measures for all 45 type pairs.

    Returns dict mapping (i,j) tuple -> {axes_align, center_align, center_axes_align, centroid_dist}.
    """
    pairwise = {}
    for i in range(N_TYPES):
        for j in range(i + 1, N_TYPES):
            mu_i = measures[i]['mu']
            mu_j = measures[j]['mu']
            V_i = measures[i]['eigvecs_top']  # [D, K]
            V_j = measures[j]['eigvecs_top']  # [D, K]

            # Centroid distance
            centroid_dist = float(np.linalg.norm(mu_i - mu_j))

            # Center alignment: cos(mu_i, mu_j)
            norm_i = np.linalg.norm(mu_i)
            norm_j = np.linalg.norm(mu_j)
            if norm_i > 0 and norm_j > 0:
                center_align = float(np.dot(mu_i, mu_j) / (norm_i * norm_j))
            else:
                center_align = 0.0

            # Axes alignment: mean |cos(PC_i^a, PC_j^b)| for top-K PCs
            # Compute |V_i^T V_j| -> [K, K], take mean
            cos_matrix = np.abs(V_i.T @ V_j)  # [K, K]
            axes_align = float(cos_matrix.mean())

            # Center-axes alignment: mean |cos(mu_i, PC_j^b)| cross-term
            # Average both directions: |mu_i . V_j| and |mu_j . V_i|
            ca_ij = np.abs(mu_i @ V_j).mean() / (norm_i + 1e-12)
            ca_ji = np.abs(mu_j @ V_i).mean() / (norm_j + 1e-12)
            center_axes_align = float((ca_ij + ca_ji) / 2.0)

            pairwise[(i, j)] = {
                'axes_align': axes_align,
                'center_align': center_align,
                'center_axes_align': center_axes_align,
                'centroid_dist': centroid_dist,
            }
    return pairwise


def compute_derived_measures(measures: dict, pairwise: dict):
    """Compute derived per-type measures from pairwise data."""
    for k in range(N_TYPES):
        # Inter-type separation: mean centroid distance from k to all others
        dists = []
        axes_aligns = []
        for i in range(N_TYPES):
            if i == k:
                continue
            pair = (min(i, k), max(i, k))
            dists.append(pairwise[pair]['centroid_dist'])
            axes_aligns.append(pairwise[pair]['axes_align'])

        inter_sep = float(np.mean(dists))
        compactness = measures[k]['R_eff'] / inter_sep if inter_sep > 0 else float('inf')
        mean_axes_align = float(np.mean(axes_aligns))

        measures[k]['inter_type_separation'] = inter_sep
        measures[k]['compactness_ratio'] = float(compactness)
        measures[k]['mean_axes_alignment'] = mean_axes_align


def bootstrap_measures(X: np.ndarray, y: np.ndarray, n_bootstrap: int = 1000,
                       rng: np.random.Generator = None):
    """Bootstrap 95% CI on per-type geometric measures."""
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)

    boot_results = {k: {'R_eff': [], 'D_eff': [], 'center_norm': [],
                         'compactness_ratio': []}
                    for k in range(N_TYPES)}

    for b in range(n_bootstrap):
        # Resample within each type
        X_boot = np.empty_like(X)
        y_boot = y.copy()
        for k in range(N_TYPES):
            mask = y == k
            idx = np.where(mask)[0]
            boot_idx = rng.choice(idx, size=len(idx), replace=True)
            X_boot[mask] = X[boot_idx]

        m = compute_per_type_measures(X_boot, y_boot)
        pw = compute_pairwise_measures(m)
        compute_derived_measures(m, pw)

        for k in range(N_TYPES):
            boot_results[k]['R_eff'].append(m[k]['R_eff'])
            boot_results[k]['D_eff'].append(m[k]['D_eff'])
            boot_results[k]['center_norm'].append(m[k]['center_norm'])
            boot_results[k]['compactness_ratio'].append(m[k]['compactness_ratio'])

    ci = {}
    for k in range(N_TYPES):
        ci[k] = {}
        for measure in ['R_eff', 'D_eff', 'center_norm', 'compactness_ratio']:
            vals = np.array(boot_results[k][measure])
            ci[k][measure] = {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals)),
                'ci_lower': float(np.percentile(vals, 2.5)),
                'ci_upper': float(np.percentile(vals, 97.5)),
            }
    return ci


def compute_confusion_from_model(X: np.ndarray, y: np.ndarray):
    """Load ARTI v3 model and compute confusion matrix on full dataset."""
    from shared.arti import ARTI, ARTIConfig

    config = ARTIConfig(encoder_dim=384, classifier_type='mlp')
    arti = ARTI(config)
    state = torch.load(str(ARTI_MODEL_PATH), weights_only=True, map_location='cpu')
    arti.load_state_dict(state)
    arti.eval()

    X_t = torch.tensor(X, dtype=torch.float32)
    batch_size = 512
    all_preds = []
    with torch.no_grad():
        for start in range(0, len(X_t), batch_size):
            batch = X_t[start:start + batch_size]
            result = arti(batch)
            all_preds.append(result['type'].cpu().numpy())

    preds = np.concatenate(all_preds)
    cm = confusion_matrix(y, preds, labels=list(range(N_TYPES)))
    return cm, preds


def compute_correlations(measures: dict, ensemble_accs: dict, pairwise: dict,
                         cm: np.ndarray):
    """Compute all correlations for success criteria evaluation."""
    # Per-type arrays (ordered by type index 0-9)
    accs = np.array([ensemble_accs[TYPE_NAMES[k]] for k in range(N_TYPES)])
    R_effs = np.array([measures[k]['R_eff'] for k in range(N_TYPES)])
    D_effs = np.array([measures[k]['D_eff'] for k in range(N_TYPES)])
    compactness = np.array([measures[k]['compactness_ratio'] for k in range(N_TYPES)])
    center_norms = np.array([measures[k]['center_norm'] for k in range(N_TYPES)])
    inter_seps = np.array([measures[k]['inter_type_separation'] for k in range(N_TYPES)])
    mean_axes = np.array([measures[k]['mean_axes_alignment'] for k in range(N_TYPES)])

    results = {}

    # SC1: rho(R_eff, accuracy) < -0.5
    rho, p = stats.spearmanr(R_effs, accs)
    r_pear, p_pear = stats.pearsonr(R_effs, accs)
    results['R_eff_vs_accuracy'] = {
        'spearman_rho': float(rho), 'spearman_p': float(p),
        'pearson_r': float(r_pear), 'pearson_p': float(p_pear),
    }

    # SC2: rho(D_eff, accuracy) < -0.4
    rho, p = stats.spearmanr(D_effs, accs)
    r_pear, p_pear = stats.pearsonr(D_effs, accs)
    results['D_eff_vs_accuracy'] = {
        'spearman_rho': float(rho), 'spearman_p': float(p),
        'pearson_r': float(r_pear), 'pearson_p': float(p_pear),
    }

    # SC3: rho(compactness, accuracy) < -0.5
    rho, p = stats.spearmanr(compactness, accs)
    r_pear, p_pear = stats.pearsonr(compactness, accs)
    results['compactness_vs_accuracy'] = {
        'spearman_rho': float(rho), 'spearman_p': float(p),
        'pearson_r': float(r_pear), 'pearson_p': float(p_pear),
    }

    # Additional correlations
    rho, p = stats.spearmanr(center_norms, accs)
    r_pear, p_pear = stats.pearsonr(center_norms, accs)
    results['center_norm_vs_accuracy'] = {
        'spearman_rho': float(rho), 'spearman_p': float(p),
        'pearson_r': float(r_pear), 'pearson_p': float(p_pear),
    }

    rho, p = stats.spearmanr(inter_seps, accs)
    r_pear, p_pear = stats.pearsonr(inter_seps, accs)
    results['inter_sep_vs_accuracy'] = {
        'spearman_rho': float(rho), 'spearman_p': float(p),
        'pearson_r': float(r_pear), 'pearson_p': float(p_pear),
    }

    rho, p = stats.spearmanr(mean_axes, accs)
    r_pear, p_pear = stats.pearsonr(mean_axes, accs)
    results['mean_axes_align_vs_accuracy'] = {
        'spearman_rho': float(rho), 'spearman_p': float(p),
        'pearson_r': float(r_pear), 'pearson_p': float(p_pear),
    }

    # SC4: rho(center_align, confusion) > 0.3
    # Collect pairwise center alignment and confusion rates
    center_aligns = []
    confusion_rates = []
    # Normalize confusion matrix to rates (row-normalized)
    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1.0)
    cm_norm = cm_norm / row_sums

    for i in range(N_TYPES):
        for j in range(i + 1, N_TYPES):
            pair = (i, j)
            center_aligns.append(pairwise[pair]['center_align'])
            # Symmetric confusion: average both directions
            conf_ij = cm_norm[i, j]
            conf_ji = cm_norm[j, i]
            confusion_rates.append((conf_ij + conf_ji) / 2.0)

    center_aligns = np.array(center_aligns)
    confusion_rates = np.array(confusion_rates)
    rho, p = stats.spearmanr(center_aligns, confusion_rates)
    r_pear, p_pear = stats.pearsonr(center_aligns, confusion_rates)
    results['center_align_vs_confusion'] = {
        'spearman_rho': float(rho), 'spearman_p': float(p),
        'pearson_r': float(r_pear), 'pearson_p': float(p_pear),
        'n_pairs': len(center_aligns),
    }

    # SC5: OLS regression R^2 > 0.60
    # 3-predictor model: R_eff + D_eff + compactness -> accuracy
    from numpy.linalg import lstsq
    X_reg = np.column_stack([R_effs, D_effs, compactness])
    X_reg_with_intercept = np.column_stack([np.ones(N_TYPES), X_reg])
    beta, residuals, rank, sv = lstsq(X_reg_with_intercept, accs, rcond=None)
    y_pred = X_reg_with_intercept @ beta
    ss_res = ((accs - y_pred) ** 2).sum()
    ss_tot = ((accs - accs.mean()) ** 2).sum()
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    # Adjusted R^2
    n = N_TYPES
    p_reg = 3  # predictors
    adj_r_squared = 1.0 - (1.0 - r_squared) * (n - 1) / (n - p_reg - 1) if n > p_reg + 1 else r_squared

    # Individual predictor R^2
    individual_r2 = {}
    for name, vals in [('R_eff', R_effs), ('D_eff', D_effs), ('compactness', compactness),
                        ('center_norm', center_norms), ('inter_sep', inter_seps),
                        ('mean_axes_align', mean_axes)]:
        X_single = np.column_stack([np.ones(N_TYPES), vals])
        b, _, _, _ = lstsq(X_single, accs, rcond=None)
        yp = X_single @ b
        ss_r = ((accs - yp) ** 2).sum()
        r2 = 1.0 - ss_r / ss_tot if ss_tot > 0 else 0.0
        individual_r2[name] = float(r2)

    results['ols_regression'] = {
        'predictors': ['R_eff', 'D_eff', 'compactness'],
        'coefficients': beta.tolist(),
        'r_squared': float(r_squared),
        'adjusted_r_squared': float(adj_r_squared),
        'individual_r2': individual_r2,
    }

    return results


def evaluate_success_criteria(correlations: dict):
    """Evaluate 5 success criteria against targets."""
    sc = {}

    # SC1: Spearman rho(R_eff, accuracy) < -0.5, p < 0.10
    rho1 = correlations['R_eff_vs_accuracy']['spearman_rho']
    p1 = correlations['R_eff_vs_accuracy']['spearman_p']
    sc['SC1_reff_predicts_accuracy'] = {
        'target': 'Spearman rho < -0.5, p < 0.10',
        'rho': rho1, 'p': p1,
        'pass': bool(rho1 < -0.5 and p1 < 0.10),
    }

    # SC2: Spearman rho(D_eff, accuracy) < -0.4, p < 0.15
    rho2 = correlations['D_eff_vs_accuracy']['spearman_rho']
    p2 = correlations['D_eff_vs_accuracy']['spearman_p']
    sc['SC2_deff_predicts_accuracy'] = {
        'target': 'Spearman rho < -0.4, p < 0.15',
        'rho': rho2, 'p': p2,
        'pass': bool(rho2 < -0.4 and p2 < 0.15),
    }

    # SC3: Spearman rho(compactness, accuracy) < -0.5, p < 0.10
    rho3 = correlations['compactness_vs_accuracy']['spearman_rho']
    p3 = correlations['compactness_vs_accuracy']['spearman_p']
    sc['SC3_compactness_predicts_accuracy'] = {
        'target': 'Spearman rho < -0.5, p < 0.10',
        'rho': rho3, 'p': p3,
        'pass': bool(rho3 < -0.5 and p3 < 0.10),
    }

    # SC4: Spearman rho(center_align, confusion) > 0.3, p < 0.10
    rho4 = correlations['center_align_vs_confusion']['spearman_rho']
    p4 = correlations['center_align_vs_confusion']['spearman_p']
    sc['SC4_center_align_predicts_confusion'] = {
        'target': 'Spearman rho > 0.3, p < 0.10',
        'rho': rho4, 'p': p4,
        'pass': bool(rho4 > 0.3 and p4 < 0.10),
    }

    # SC5: OLS R^2 > 0.60
    r2 = correlations['ols_regression']['r_squared']
    adj_r2 = correlations['ols_regression']['adjusted_r_squared']
    sc['SC5_combined_model_r2'] = {
        'target': 'OLS R^2 > 0.60',
        'r_squared': r2, 'adjusted_r_squared': adj_r2,
        'pass': bool(r2 > 0.60),
    }

    n_pass = sum(1 for v in sc.values() if v['pass'])
    sc['n_pass'] = n_pass
    sc['n_total'] = 5
    return sc


# ===================================================================
# Main
# ===================================================================

def main():
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("E19: Manifold Geometry of Reasoning Types — GCMC-Inspired Analysis")
    logger.info("=" * 70)

    # Create output directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ----- Load data -----
    logger.info("Loading ARTI v3 dataset...")
    data = torch.load(str(DATASET_PATH), weights_only=False)
    X = data['embeddings'].numpy()  # [7500, 384]
    y = data['labels'].numpy()      # [7500]
    logger.info(f"  Dataset: {X.shape[0]} samples, {X.shape[1]}D, {N_TYPES} types")
    for k in range(N_TYPES):
        logger.info(f"  Type {k} ({TYPE_NAMES[k]}): n={(y == k).sum()}")

    # ----- Load ensemble accuracies -----
    logger.info("Loading ensemble results...")
    with open(str(ENSEMBLE_RESULTS_PATH)) as f:
        ens = json.load(f)
    ensemble_accs = {name: ens['per_type'][name]['ensemble_acc']
                     for name in TYPE_NAMES}
    logger.info(f"  Ensemble accuracy: {ens['ensemble_accuracy']:.1%}")
    for name in TYPE_NAMES:
        logger.info(f"    {name:12s}: {ensemble_accs[name]:.1%}")

    # ----- Step 1: Per-type geometric measures -----
    logger.info("-" * 50)
    logger.info("Step 1: Computing per-type geometric measures...")
    measures = compute_per_type_measures(X, y)

    logger.info(f"  {'Type':15s}  {'R_eff':>7s}  {'D_eff':>7s}  {'Center':>7s}  {'Acc':>6s}")
    for k in range(N_TYPES):
        m = measures[k]
        acc = ensemble_accs[TYPE_NAMES[k]]
        logger.info(f"  {TYPE_FULL_NAMES[k]:15s}  {m['R_eff']:7.4f}  {m['D_eff']:7.2f}  "
                     f"{m['center_norm']:7.4f}  {acc:5.1%}")

    # ----- Step 2: Pairwise measures -----
    logger.info("-" * 50)
    logger.info("Step 2: Computing pairwise measures (45 pairs)...")
    pairwise = compute_pairwise_measures(measures)

    # Log a few example pairs
    for (i, j), pw in list(pairwise.items())[:5]:
        logger.info(f"  ({TYPE_NAMES[i]:10s}, {TYPE_NAMES[j]:10s}): "
                     f"ctr_align={pw['center_align']:.3f}  "
                     f"axes_align={pw['axes_align']:.3f}  "
                     f"dist={pw['centroid_dist']:.3f}")

    # ----- Step 3: Derived measures -----
    logger.info("-" * 50)
    logger.info("Step 3: Computing derived per-type measures...")
    compute_derived_measures(measures, pairwise)

    logger.info(f"  {'Type':15s}  {'Separation':>10s}  {'Compactness':>11s}  {'MeanAxes':>8s}")
    for k in range(N_TYPES):
        m = measures[k]
        logger.info(f"  {TYPE_FULL_NAMES[k]:15s}  {m['inter_type_separation']:10.4f}  "
                     f"{m['compactness_ratio']:11.4f}  {m['mean_axes_alignment']:8.4f}")

    # ----- Step 4: Confusion matrix from ARTI v3 -----
    logger.info("-" * 50)
    logger.info("Step 4: Computing confusion matrix from ARTI v3 model...")
    cm, preds = compute_confusion_from_model(X, y)
    model_acc = (preds == y).mean()
    logger.info(f"  ARTI v3 accuracy on full dataset: {model_acc:.1%}")

    # ----- Step 5: Correlations -----
    logger.info("-" * 50)
    logger.info("Step 5: Computing correlations with ARTI accuracy...")
    correlations = compute_correlations(measures, ensemble_accs, pairwise, cm)

    for name, vals in correlations.items():
        if name == 'ols_regression':
            logger.info(f"  OLS: R^2={vals['r_squared']:.3f}  adj_R^2={vals['adjusted_r_squared']:.3f}")
            for pred_name, r2 in vals['individual_r2'].items():
                logger.info(f"    {pred_name}: R^2={r2:.3f}")
        else:
            logger.info(f"  {name}: rho={vals['spearman_rho']:.3f}  p={vals['spearman_p']:.4f}  "
                         f"pearson_r={vals['pearson_r']:.3f}")

    # ----- Step 6: Bootstrap CIs -----
    logger.info("-" * 50)
    logger.info(f"Step 6: Bootstrap ({N_BOOTSTRAP} resamples) for 95% CIs...")
    rng = np.random.default_rng(RANDOM_SEED)
    bootstrap_ci = bootstrap_measures(X, y, N_BOOTSTRAP, rng)

    for k in range(N_TYPES):
        ci = bootstrap_ci[k]
        logger.info(f"  {TYPE_FULL_NAMES[k]:15s}  "
                     f"R_eff=[{ci['R_eff']['ci_lower']:.4f}, {ci['R_eff']['ci_upper']:.4f}]  "
                     f"D_eff=[{ci['D_eff']['ci_lower']:.1f}, {ci['D_eff']['ci_upper']:.1f}]")

    # ----- Step 7: Success criteria -----
    logger.info("-" * 50)
    logger.info("Step 7: Evaluating success criteria...")
    sc = evaluate_success_criteria(correlations)

    for sc_name, sc_val in sc.items():
        if isinstance(sc_val, dict):
            status = "PASS" if sc_val['pass'] else "FAIL"
            logger.info(f"  {sc_name}: {status} — {sc_val['target']}")
            if 'rho' in sc_val:
                logger.info(f"    rho={sc_val['rho']:.3f}, p={sc_val['p']:.4f}")
            if 'r_squared' in sc_val:
                logger.info(f"    R^2={sc_val['r_squared']:.3f}")

    logger.info(f"\n  SUCCESS CRITERIA: {sc['n_pass']}/{sc['n_total']} met")

    # ----- Serialize results -----
    elapsed = time.time() - t0

    # Prepare serializable per-type measures
    per_type_serializable = {}
    for k in range(N_TYPES):
        m = measures[k]
        per_type_serializable[TYPE_NAMES[k]] = {
            'type_id': k,
            'full_name': TYPE_FULL_NAMES[k],
            'n_samples': m['n_samples'],
            'R_eff': m['R_eff'],
            'D_eff': m['D_eff'],
            'center_norm': m['center_norm'],
            'trace_cov': m['trace_cov'],
            'top_eigvals': m['top_eigvals'],
            'inter_type_separation': m['inter_type_separation'],
            'compactness_ratio': m['compactness_ratio'],
            'mean_axes_alignment': m['mean_axes_alignment'],
            'ensemble_accuracy': ensemble_accs[TYPE_NAMES[k]],
        }

    # Prepare serializable pairwise measures
    pairwise_serializable = {}
    for (i, j), pw in pairwise.items():
        key = f"{TYPE_NAMES[i]}__{TYPE_NAMES[j]}"
        pairwise_serializable[key] = pw

    # Prepare serializable bootstrap CIs
    bootstrap_serializable = {}
    for k in range(N_TYPES):
        bootstrap_serializable[TYPE_NAMES[k]] = bootstrap_ci[k]

    # Build 10x10 matrices for heatmap
    center_align_matrix = np.zeros((N_TYPES, N_TYPES))
    axes_align_matrix = np.zeros((N_TYPES, N_TYPES))
    for i in range(N_TYPES):
        center_align_matrix[i, i] = 1.0  # self-alignment = 1
        axes_align_matrix[i, i] = 1.0
        for j in range(i + 1, N_TYPES):
            ca = pairwise[(i, j)]['center_align']
            aa = pairwise[(i, j)]['axes_align']
            center_align_matrix[i, j] = ca
            center_align_matrix[j, i] = ca
            axes_align_matrix[i, j] = aa
            axes_align_matrix[j, i] = aa

    output = {
        'experiment': 'E19_manifold_geometry',
        'description': 'GCMC-inspired geometric analysis of 10 reasoning type manifolds',
        'hypothesis': 'Geometric properties that predict task-efficiency in biological neural '
                      'manifolds also predict ARTI classification accuracy in artificial '
                      'reasoning manifolds',
        'reference': 'Chou et al. 2024, GCMC framework [26]',
        'dataset': {
            'path': str(DATASET_PATH),
            'n_samples': int(X.shape[0]),
            'embedding_dim': int(X.shape[1]),
            'n_types': N_TYPES,
            'type_names': TYPE_NAMES,
        },
        'per_type_measures': per_type_serializable,
        'pairwise_measures': pairwise_serializable,
        'center_align_matrix': center_align_matrix.tolist(),
        'axes_align_matrix': axes_align_matrix.tolist(),
        'confusion_matrix': cm.tolist(),
        'arti_v3_full_accuracy': float(model_acc),
        'correlations': correlations,
        'bootstrap_ci': bootstrap_serializable,
        'success_criteria': sc,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'elapsed_seconds': round(elapsed, 1),
    }

    out_path = RESULTS_DIR / "e19_results.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"\nResults saved to {out_path}")

    logger.info("=" * 70)
    logger.info(f"E19 complete in {elapsed:.1f}s — {sc['n_pass']}/{sc['n_total']} success criteria met")
    logger.info("=" * 70)

    return output


if __name__ == '__main__':
    main()
