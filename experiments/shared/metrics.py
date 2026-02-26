#!/usr/bin/env python3
"""
Metrics for CO-FRN Experiments.

Adapted from Paper 11 E1 metrics.py and extended with:
- Centered redundancy (rho) — from Paper 11
- Pareto frontier computation — for E1
- Transfer ratio — for E2
- Sample efficiency ratio — for E2
- Scaling curve fitting — for E3
- Anchor utilization metrics — for E4
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from itertools import combinations
from dataclasses import dataclass
from scipy.spatial.distance import cdist
from scipy.optimize import curve_fit


# ─── Redundancy Metrics (from Paper 11 E1) ────────────────────────────────

@dataclass
class RedundancyResult:
    """Result of redundancy computation for a single pattern."""
    pattern: str
    rho_centroid: float
    rho_mmd: float
    pairwise_cosines: Dict[Tuple[str, str], float]
    domain_centroids: Dict[str, np.ndarray]
    n_samples_per_domain: Dict[str, int]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def compute_mmd_squared(
    X: np.ndarray,
    Y: np.ndarray,
    sigma: Optional[float] = None,
    max_samples: int = 2000,
) -> float:
    """Unbiased MMD^2 estimator with Gaussian kernel (vectorized)."""
    m, n = len(X), len(Y)
    if m < 2 or n < 2:
        return 0.0

    # Subsample if too large for O(n^2) kernel matrix
    if m > max_samples:
        idx = np.random.choice(m, max_samples, replace=False)
        X = X[idx]
        m = max_samples
    if n > max_samples:
        idx = np.random.choice(n, max_samples, replace=False)
        Y = Y[idx]
        n = max_samples

    if sigma is None:
        # Estimate sigma from a subsample to avoid huge distance matrix
        sub = min(500, m + n)
        XY = np.vstack([X, Y])
        idx = np.random.choice(len(XY), sub, replace=False) if len(XY) > sub else np.arange(len(XY))
        sub_dists = cdist(XY[idx], XY[idx], metric='euclidean')
        nonzero = sub_dists[sub_dists > 0]
        sigma = float(np.median(nonzero)) if len(nonzero) > 0 else 1.0

    gamma = -1.0 / (2 * sigma ** 2)

    # Vectorized kernel matrices
    K_xx = np.exp(gamma * cdist(X, X, metric='sqeuclidean'))
    K_yy = np.exp(gamma * cdist(Y, Y, metric='sqeuclidean'))
    K_xy = np.exp(gamma * cdist(X, Y, metric='sqeuclidean'))

    # Unbiased: exclude diagonal for K_xx and K_yy
    np.fill_diagonal(K_xx, 0.0)
    np.fill_diagonal(K_yy, 0.0)

    return (K_xx.sum() / (m * (m - 1)) +
            K_yy.sum() / (n * (n - 1)) -
            2 * K_xy.sum() / (m * n))


def compute_centered_redundancy(
    embeddings_by_domain: Dict[str, np.ndarray],
    pattern_name: str = "unknown",
) -> RedundancyResult:
    """
    Compute centered redundancy rho.

    Centers embeddings by global mean to address anisotropy in
    LLM embeddings where all vectors have high cosine similarity.

    rho = 1 - mean(pairwise cosine similarities of domain centroids)

    Args:
        embeddings_by_domain: {domain: [n_samples, dim]} arrays
        pattern_name: identifier

    Returns:
        RedundancyResult
    """
    # Center by global mean
    all_embs = np.vstack(list(embeddings_by_domain.values()))
    global_mean = np.mean(all_embs, axis=0)

    centered = {d: embs - global_mean for d, embs in embeddings_by_domain.items()}

    # Compute centroids
    centroids = {d: np.mean(e, axis=0) for d, e in centered.items()}

    # Pairwise cosines
    domains = list(centroids.keys())
    pairwise_cos = {}
    for d1, d2 in combinations(domains, 2):
        pairwise_cos[(d1, d2)] = cosine_similarity(centroids[d1], centroids[d2])

    rho = 1.0 - np.mean(list(pairwise_cos.values())) if pairwise_cos else 0.0

    # MMD
    mmd_vals = []
    for d1, d2 in combinations(domains, 2):
        mmd_vals.append(compute_mmd_squared(centered[d1], centered[d2]))
    rho_mmd = float(np.mean(mmd_vals)) if mmd_vals else 0.0

    return RedundancyResult(
        pattern=pattern_name,
        rho_centroid=rho,
        rho_mmd=rho_mmd,
        pairwise_cosines=pairwise_cos,
        domain_centroids=centroids,
        n_samples_per_domain={d: len(e) for d, e in embeddings_by_domain.items()},
    )


def bootstrap_rho_ci(
    embeddings_by_domain: Dict[str, np.ndarray],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap 95% CI for rho."""
    rng = np.random.RandomState(seed)
    rho_samples = []

    for _ in range(n_bootstrap):
        resampled = {}
        for d, embs in embeddings_by_domain.items():
            idx = rng.choice(len(embs), size=len(embs), replace=True)
            resampled[d] = embs[idx]
        result = compute_centered_redundancy(resampled)
        rho_samples.append(result.rho_centroid)

    rho_samples = np.array(rho_samples)
    alpha = 1 - confidence
    return (
        float(np.mean(rho_samples)),
        float(np.percentile(rho_samples, 100 * alpha / 2)),
        float(np.percentile(rho_samples, 100 * (1 - alpha / 2))),
    )


# ─── Pareto Frontier (E1) ─────────────────────────────────────────────────

@dataclass
class ParetoPoint:
    """A single point in the Pareto analysis."""
    method: str
    params: int
    accuracy: float
    flops: Optional[float] = None
    is_pareto: bool = False


def compute_pareto_frontier(
    points: List[ParetoPoint],
    maximize_y: bool = True,
) -> List[ParetoPoint]:
    """
    Compute Pareto frontier from a set of points.

    A point is Pareto-optimal if no other point has both:
    - fewer parameters (x-axis, minimize)
    - higher accuracy (y-axis, maximize)

    Args:
        points: list of ParetoPoint
        maximize_y: if True, higher y is better

    Returns:
        Same list with is_pareto flags set
    """
    # Sort by params ascending
    sorted_pts = sorted(enumerate(points), key=lambda x: x[1].params)

    if maximize_y:
        best_y = float('-inf')
        for orig_idx, pt in sorted_pts:
            if pt.accuracy >= best_y:
                points[orig_idx].is_pareto = True
                best_y = pt.accuracy
    else:
        best_y = float('inf')
        for orig_idx, pt in sorted_pts:
            if pt.accuracy <= best_y:
                points[orig_idx].is_pareto = True
                best_y = pt.accuracy

    return points


def pareto_dominance_count(
    target: ParetoPoint,
    others: List[ParetoPoint],
) -> int:
    """Count how many points the target dominates."""
    count = 0
    for other in others:
        if (target.params <= other.params and target.accuracy >= other.accuracy
                and (target.params < other.params or target.accuracy > other.accuracy)):
            count += 1
    return count


# ─── Transfer Ratio (E2) ──────────────────────────────────────────────────

def transfer_ratio(
    source_accuracy: float,
    target_accuracy: float,
) -> float:
    """
    Compute transfer ratio: target_acc / source_acc.

    Measures how well performance transfers to a new domain.
    Values > 0.7 = near-domain, > 0.3 = meaningful far-domain.
    """
    if source_accuracy == 0:
        return 0.0
    return target_accuracy / source_accuracy


def sample_efficiency_ratio(
    cofrn_curve: Dict[int, float],
    baseline_curve: Dict[int, float],
    reference_n: int = 64,
) -> float:
    """
    Compute sample efficiency ratio.

    How many examples does the baseline need to match CO-FRN at `reference_n`?
    SER = baseline_n_to_match / reference_n

    Args:
        cofrn_curve: {n_examples: accuracy} for CO-FRN
        baseline_curve: {n_examples: accuracy} for baseline
        reference_n: CO-FRN data point to match

    Returns:
        ratio >= 1.0 (higher = CO-FRN more sample-efficient)
    """
    cofrn_acc = cofrn_curve.get(reference_n, 0.0)

    # Find smallest n where baseline reaches cofrn_acc
    sorted_ns = sorted(baseline_curve.keys())
    for n in sorted_ns:
        if baseline_curve[n] >= cofrn_acc:
            return n / reference_n

    # Baseline never reaches CO-FRN accuracy
    return float('inf')


def adaptation_curve_auc(
    curve: Dict[int, float],
    max_n: int = 1024,
) -> float:
    """
    Area under the adaptation curve (normalized).

    Higher AUC = better few-shot adaptation.
    """
    sorted_points = sorted(curve.items())
    if len(sorted_points) < 2:
        return 0.0

    # Trapezoidal integration
    auc = 0.0
    for i in range(len(sorted_points) - 1):
        n1, acc1 = sorted_points[i]
        n2, acc2 = sorted_points[i + 1]
        # Use log scale for x
        auc += (np.log(n2) - np.log(n1)) * (acc1 + acc2) / 2

    # Normalize by total x range
    total_range = np.log(sorted_points[-1][0]) - np.log(sorted_points[0][0])
    return auc / total_range if total_range > 0 else 0.0


# ─── Scaling Curves (E3) ──────────────────────────────────────────────────

def log_linear_fit(
    params: np.ndarray,
    accuracies: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Fit log-linear model: accuracy = a * log10(params) + b

    Returns:
        (slope_a, intercept_b, r_squared)
    """
    log_params = np.log10(params)

    def model(x, a, b):
        return a * x + b

    try:
        popt, _ = curve_fit(model, log_params, accuracies)
        a, b = popt
        predicted = model(log_params, a, b)
        ss_res = np.sum((accuracies - predicted) ** 2)
        ss_tot = np.sum((accuracies - np.mean(accuracies)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return float(a), float(b), float(r2)
    except Exception:
        return 0.0, float(np.mean(accuracies)), 0.0


def detect_saturation(
    params: np.ndarray,
    accuracies: np.ndarray,
    threshold: float = 0.01,
) -> Optional[float]:
    """
    Detect parameter budget where accuracy saturates.

    Saturation = accuracy(2*budget) - accuracy(budget) < threshold.

    Returns:
        Saturation budget (params) or None if not saturated
    """
    sorted_indices = np.argsort(params)
    sorted_params = params[sorted_indices]
    sorted_acc = accuracies[sorted_indices]

    for i in range(len(sorted_params) - 1):
        gain = sorted_acc[i + 1] - sorted_acc[i]
        if gain < threshold:
            return float(sorted_params[i])

    return None


def bootstrap_slope_ci(
    params: np.ndarray,
    accuracies: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap CI for the scaling slope."""
    rng = np.random.RandomState(seed)
    slopes = []

    for _ in range(n_bootstrap):
        idx = rng.choice(len(params), size=len(params), replace=True)
        a, _, _ = log_linear_fit(params[idx], accuracies[idx])
        slopes.append(a)

    slopes = np.array(slopes)
    alpha = 1 - confidence
    return (
        float(np.mean(slopes)),
        float(np.percentile(slopes, 100 * alpha / 2)),
        float(np.percentile(slopes, 100 * (1 - alpha / 2))),
    )


# ─── Anchor/Manifold Metrics (E4) ─────────────────────────────────────────

def anchor_utilization_entropy(
    weights: np.ndarray,
    n_anchors: int,
) -> Dict[str, float]:
    """
    Compute anchor utilization metrics from attention weights.

    Args:
        weights: [N, n_anchors] attention weights over dataset
        n_anchors: number of anchors

    Returns:
        Dict with entropy, normalized_entropy, participation_ratio
    """
    mean_w = np.mean(weights, axis=0)
    mean_w = mean_w / (np.sum(mean_w) + 1e-10)

    entropy = -np.sum(mean_w * np.log(mean_w + 1e-10))
    max_entropy = np.log(n_anchors)
    norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    # Participation ratio: (sum w)^2 / sum(w^2)
    pr = np.sum(mean_w) ** 2 / (np.sum(mean_w ** 2) + 1e-10)

    return {
        'entropy': float(entropy),
        'normalized_entropy': float(norm_entropy),
        'participation_ratio': float(pr),
        'max_weight': float(np.max(mean_w)),
        'min_weight': float(np.min(mean_w)),
    }


def per_domain_conditional_entropy(
    weights: np.ndarray,
    domains: np.ndarray,
    n_anchors: int,
) -> Dict[str, float]:
    """
    Compute per-domain conditional entropy of anchor assignments.

    Low entropy per domain = local collapse (bad).
    Target: participation ratio > 3 per domain.

    Args:
        weights: [N, n_anchors] attention weights
        domains: [N] domain labels
        n_anchors: number of anchors

    Returns:
        Dict mapping domain -> {entropy, participation_ratio}
    """
    unique_domains = np.unique(domains)
    results = {}

    for d in unique_domains:
        mask = domains == d
        domain_weights = weights[mask]

        if len(domain_weights) == 0:
            continue

        metrics = anchor_utilization_entropy(domain_weights, n_anchors)
        results[str(d)] = metrics

    return results


def nmi_score(
    labels_a: np.ndarray,
    labels_b: np.ndarray,
) -> float:
    """Normalized Mutual Information between two label sets."""
    from sklearn.metrics import normalized_mutual_info_score
    return float(normalized_mutual_info_score(labels_a, labels_b))


# ─── Cross-Domain Divergence (E5) ────────────────────────────────────

def cross_domain_js_divergence(
    weights: np.ndarray,
    domains: np.ndarray,
) -> Dict[str, float]:
    """
    Pairwise Jensen-Shannon divergence of anchor weight distributions between domains.

    For each pair of domains, computes JS divergence between the mean anchor
    weight distributions. Low JS divergence = domain-invariant operators.

    Args:
        weights: [N, n_anchors] attention weights over dataset
        domains: [N] domain labels (strings or ints)

    Returns:
        Dict with:
            - 'pairwise': {(d1, d2): js_div} for all domain pairs
            - 'mean_js': average JS divergence across all pairs
            - 'n_below_threshold': count of pairs with JS < 0.5
            - 'n_pairs': total number of domain pairs
    """
    unique_domains = sorted(set(domains))

    # Compute mean distribution per domain
    domain_dists = {}
    for d in unique_domains:
        mask = np.array([dd == d for dd in domains])
        if mask.sum() == 0:
            continue
        mean_w = np.mean(weights[mask], axis=0)
        mean_w = mean_w / (np.sum(mean_w) + 1e-10)  # normalize
        domain_dists[d] = mean_w

    # Pairwise JS divergence
    pairwise = {}
    domains_list = list(domain_dists.keys())
    for i in range(len(domains_list)):
        for j in range(i + 1, len(domains_list)):
            d1, d2 = domains_list[i], domains_list[j]
            p, q = domain_dists[d1], domain_dists[d2]
            # JS divergence = 0.5 * KL(p||m) + 0.5 * KL(q||m) where m = (p+q)/2
            m = 0.5 * (p + q)
            kl_pm = np.sum(p * np.log((p + 1e-10) / (m + 1e-10)))
            kl_qm = np.sum(q * np.log((q + 1e-10) / (m + 1e-10)))
            js = 0.5 * kl_pm + 0.5 * kl_qm
            pairwise[f"{d1}_vs_{d2}"] = float(js)

    js_values = list(pairwise.values())
    return {
        'pairwise': pairwise,
        'mean_js': float(np.mean(js_values)) if js_values else 0.0,
        'n_below_threshold': sum(1 for v in js_values if v < 0.5),
        'n_pairs': len(js_values),
    }


# ─── Statistical Tests ────────────────────────────────────────────────────

def paired_permutation_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_permutations: int = 10000,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Paired permutation test for difference in means.

    Tests H0: mean(A) = mean(B).

    Returns:
        (observed_diff, p_value)
    """
    rng = np.random.RandomState(seed)
    observed_diff = np.mean(scores_a) - np.mean(scores_b)
    diffs = scores_a - scores_b

    count = 0
    for _ in range(n_permutations):
        signs = rng.choice([-1, 1], size=len(diffs))
        perm_diff = np.mean(diffs * signs)
        if abs(perm_diff) >= abs(observed_diff):
            count += 1

    p_value = (count + 1) / (n_permutations + 1)
    return float(observed_diff), float(p_value)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Cohen's d effect size."""
    n_a, n_b = len(a), len(b)
    var_a, var_b = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled_std == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_std)


if __name__ == "__main__":
    print("Testing metrics...")

    # Redundancy
    np.random.seed(42)
    embs = {
        'math': np.random.randn(50, 256) + np.array([1, 0] + [0] * 254),
        'science': np.random.randn(50, 256) + np.array([0, 1] + [0] * 254),
        'logic': np.random.randn(50, 256) + np.array([-1, 0] + [0] * 254),
    }
    rho = compute_centered_redundancy(embs, "test")
    print(f"Centered rho: {rho.rho_centroid:.4f}")

    # Bootstrap
    mean, lo, hi = bootstrap_rho_ci(embs, n_bootstrap=100)
    print(f"Bootstrap 95% CI: [{lo:.4f}, {hi:.4f}]")

    # Pareto frontier
    points = [
        ParetoPoint('linear', 3000, 0.25),
        ParetoPoint('mlp', 200000, 0.45),
        ParetoPoint('lora', 590000, 0.52),
        ParetoPoint('cofrn', 726000, 0.58),
        ParetoPoint('full_ft', 124000000, 0.62),
    ]
    compute_pareto_frontier(points)
    pareto = [p for p in points if p.is_pareto]
    print(f"\nPareto frontier: {[p.method for p in pareto]}")

    # Transfer ratio
    tr = transfer_ratio(0.58, 0.42)
    print(f"\nTransfer ratio: {tr:.2f}")

    # Sample efficiency
    cofrn = {16: 0.30, 64: 0.45, 256: 0.52, 1024: 0.55}
    baseline = {16: 0.20, 64: 0.30, 256: 0.42, 1024: 0.48}
    ser = sample_efficiency_ratio(cofrn, baseline, reference_n=64)
    print(f"Sample efficiency ratio: {ser:.1f}x")

    # Scaling fit
    params = np.array([50000, 100000, 250000, 500000, 1000000, 5000000])
    accs = np.array([0.30, 0.38, 0.45, 0.50, 0.53, 0.55])
    a, b, r2 = log_linear_fit(params, accs)
    print(f"\nScaling fit: acc = {a:.4f} * log10(params) + {b:.4f}, R²={r2:.4f}")

    sat = detect_saturation(params, accs)
    print(f"Saturation at: {sat:,.0f} params" if sat else "No saturation detected")

    # Anchor utilization
    weights = np.random.dirichlet(np.ones(16), size=100)
    util = anchor_utilization_entropy(weights, 16)
    print(f"\nAnchor utilization: entropy={util['entropy']:.3f}, "
          f"norm={util['normalized_entropy']:.3f}, PR={util['participation_ratio']:.2f}")

    # Permutation test
    a = np.random.randn(50) + 0.3
    b = np.random.randn(50)
    diff, p = paired_permutation_test(a, b, n_permutations=1000)
    d = cohens_d(a, b)
    print(f"\nPermutation test: diff={diff:.3f}, p={p:.4f}, d={d:.3f}")

    print("\nAll metrics tests passed!")
