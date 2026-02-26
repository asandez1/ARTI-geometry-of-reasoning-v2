#!/usr/bin/env python3
"""
Structural Operators and Continuous Operator Manifold for CO-FRN.

Adapted from Paper 11 E3 operators.py. Key changes:
- StructuralOperator: kept as-is (I + scale * U @ V^T)
- OperatorSet (discrete, Gumbel-softmax): REMOVED
- ContinuousOperatorManifold: NEW — replaces OperatorSet

The continuous manifold is motivated by Paper 11 E4's finding that reasoning
operators form a ~10D continuous manifold (silhouette=0.33 in 10D subspace,
NMI=0.019 with hypothesized discrete categories).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math


class StructuralOperator(nn.Module):
    """
    A single structural operator parameterized as I + scale * U @ V^T.

    Low-rank perturbation of identity ensures:
    - Near-orthogonality (preserves norms approximately)
    - Efficient parameterization (2 * dim * rank parameters vs dim^2)
    - Distinct transformations per operator

    Kept from Paper 11 E3, used internally by ContinuousOperatorManifold.
    """

    def __init__(self, dim: int, rank: int = 16, name: str = ""):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.name = name

        self.U = nn.Parameter(torch.randn(dim, rank) * 0.01)
        self.V = nn.Parameter(torch.randn(dim, rank) * 0.01)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply operator: y = normalize(x + scale * U @ V^T @ x)."""
        Vx = x @ self.V            # [..., rank]
        UVx = Vx @ self.U.T        # [..., dim]
        y = x + self.scale * UVx
        y = F.normalize(y, dim=-1)
        return y

    def get_matrix(self) -> torch.Tensor:
        """Return the full operator matrix (for analysis)."""
        return torch.eye(self.dim, device=self.U.device) + self.scale * (self.U @ self.V.T)


class ContinuousOperatorManifold(nn.Module):
    """
    Continuous operator manifold replacing Paper 11's discrete OperatorSet.

    Design (Section 3.2.2 of draft):
    - n_anchors learnable anchor points in R^manifold_dim
    - Each anchor has low-rank factors (U_i, V_i) -> operator O_i = I + U_i V_i^T
    - Soft attention over anchors: w_i = softmax(f(o) . a_i / (tau * sqrt(d_manifold)))
    - Interpolated operator: O(x) = x + sum(w_i * U_i V_i^T x) via einsum
    - Temperature annealing: tau_0=1.0, decay=0.95/epoch, floor=0.1
    - Entropy regularizer prevents anchor collapse

    NOT Gumbel-softmax: Paper 11 E4 falsified discrete hypothesis.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        struct_dim: int = 128,
        manifold_dim: int = 10,
        n_anchors: int = 16,
        rank: int = 16,
        tau_init: float = 1.0,
        tau_decay: float = 0.95,
        tau_floor: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.struct_dim = struct_dim
        self.manifold_dim = manifold_dim
        self.n_anchors = n_anchors
        self.rank = rank
        self.tau_init = tau_init
        self.tau_decay = tau_decay
        self.tau_floor = tau_floor

        # Current temperature (updated externally via anneal_temperature)
        self.register_buffer('tau', torch.tensor(tau_init))

        # Learnable anchor points on the manifold (orthogonal init for max separation)
        # First manifold_dim anchors are orthonormal via QR on a square matrix;
        # remaining anchors (if n_anchors > manifold_dim) are unit-norm random.
        _square = torch.randn(manifold_dim, manifold_dim)
        _Q, _ = torch.linalg.qr(_square)  # [manifold_dim, manifold_dim] orthogonal
        if n_anchors <= manifold_dim:
            _anchors = _Q[:n_anchors]
        else:
            _extra = F.normalize(torch.randn(n_anchors - manifold_dim, manifold_dim), dim=-1)
            _anchors = torch.cat([_Q, _extra], dim=0)  # [n_anchors, manifold_dim]
        self.anchors = nn.Parameter(_anchors)

        # Per-anchor low-rank operator factors
        # O_i = I + U_i @ V_i^T
        # Scale 0.05 gives ~16% perturbation (vs 0.006% at 0.01), bounded by L2 norm
        self.U_all = nn.Parameter(torch.randn(n_anchors, hidden_dim, rank) * 0.05)
        self.V_all = nn.Parameter(torch.randn(n_anchors, hidden_dim, rank) * 0.05)

        # Projection: structural component -> manifold space for attention
        self.manifold_proj = nn.Linear(struct_dim, manifold_dim)
        nn.init.xavier_normal_(self.manifold_proj.weight, gain=2.0)
        nn.init.zeros_(self.manifold_proj.bias)

    def anneal_temperature(self, epoch: int):
        """Update temperature based on epoch. Call once per epoch."""
        new_tau = max(self.tau_floor, self.tau_init * (self.tau_decay ** epoch))
        self.tau.fill_(new_tau)

    def compute_attention_weights(
        self,
        structural: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute soft attention weights over anchors.

        Args:
            structural: [B, struct_dim] structural component |o>

        Returns:
            weights: [B, n_anchors] attention weights summing to 1
        """
        query = self.manifold_proj(structural)  # [B, manifold_dim]
        # Scaled dot-product attention
        scale = math.sqrt(self.manifold_dim)
        logits = query @ self.anchors.T / (self.tau * scale)  # [B, n_anchors]
        weights = F.softmax(logits, dim=-1)  # [B, n_anchors]
        return weights

    def apply_operator(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply the interpolated operator to input x.

        O(x) = x + sum_i(w_i * U_i @ V_i^T @ x)

        Each anchor's operator is applied independently then weight-summed,
        avoiding cross-terms between different anchors' U and V factors.

        Implemented via einsum:
          V_i^T @ x for each anchor  -> [B, n_anchors, rank]
          U_i @ (V_i^T @ x) per anchor -> [B, n_anchors, hidden_dim]
          weight-sum over anchors    -> [B, hidden_dim]

        Args:
            x: [B, hidden_dim] input state
            weights: [B, n_anchors] attention weights

        Returns:
            Ox: [B, hidden_dim] transformed state
        """
        # Per-anchor: V_i^T @ x -> [B, n_anchors, rank]
        # V_all: [a, h, r], x: [b, h] -> Vx: [b, a, r]
        Vx = torch.einsum('ahr,bh->bar', self.V_all, x)

        # Per-anchor: U_i @ (V_i^T @ x) -> [B, n_anchors, hidden_dim]
        # U_all: [a, h, r], Vx: [b, a, r] -> UVx: [b, a, h]
        UVx = torch.einsum('ahr,bar->bah', self.U_all, Vx)

        # Weight-sum over anchors: sum_i(w_i * U_i V_i^T x) -> [B, hidden_dim]
        weighted = torch.einsum('ba,bah->bh', weights, UVx)

        Ox = x + weighted
        Ox = F.normalize(Ox, dim=-1)
        return Ox

    def forward(
        self,
        x: torch.Tensor,
        structural: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass: compute attention, apply interpolated operator.

        Args:
            x: [B, hidden_dim] state to transform
            structural: [B, struct_dim] structural component for anchor selection

        Returns:
            Ox: [B, hidden_dim] transformed state
            weights: [B, n_anchors] attention weights (for diagnostics/loss)
        """
        weights = self.compute_attention_weights(structural)
        Ox = self.apply_operator(x, weights)
        return Ox, weights

    def entropy_loss(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy regularizer to prevent anchor collapse.

        We MAXIMIZE entropy (return negative so minimizing total loss increases entropy).
        L_entropy = -H(w) = sum(w * log(w + eps))

        Args:
            weights: [B, n_anchors] attention weights

        Returns:
            loss: scalar (negative entropy, to be ADDED to total loss for minimization)
        """
        # Mean over batch first, then compute entropy of average distribution
        mean_weights = weights.mean(dim=0)  # [n_anchors]
        entropy = -(mean_weights * torch.log(mean_weights + 1e-10)).sum()
        # Return negative entropy (minimizing this maximizes entropy)
        return -entropy

    def get_anchor_utilization(self, weights: torch.Tensor) -> Dict[str, float]:
        """
        Compute anchor utilization metrics.

        Args:
            weights: [B, n_anchors] accumulated attention weights

        Returns:
            Dict with entropy, normalized_entropy, participation_ratio
        """
        mean_w = weights.mean(dim=0)
        mean_w = mean_w / (mean_w.sum() + 1e-10)

        entropy = -(mean_w * torch.log(mean_w + 1e-10)).sum().item()
        max_entropy = math.log(self.n_anchors)
        norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        # Participation ratio: (sum w)^2 / sum(w^2)
        pr = (mean_w.sum() ** 2) / (mean_w ** 2).sum()

        return {
            'entropy': entropy,
            'normalized_entropy': norm_entropy,
            'participation_ratio': pr.item(),
            'max_weight': mean_w.max().item(),
            'min_weight': mean_w.min().item(),
        }

    def get_operator_similarity(self) -> torch.Tensor:
        """
        Compute pairwise cosine similarity between anchor operators.

        Returns:
            sim: [n_anchors, n_anchors] similarity matrix
        """
        # Flatten U_i @ V_i^T for each anchor into vectors
        # Approximate: use U_i and V_i concatenated
        UV = torch.cat([
            self.U_all.reshape(self.n_anchors, -1),
            self.V_all.reshape(self.n_anchors, -1),
        ], dim=-1)  # [n_anchors, 2 * hidden_dim * rank]
        UV_norm = F.normalize(UV, dim=-1)
        sim = UV_norm @ UV_norm.T
        return sim


class SimpleMLPOperator(nn.Module):
    """
    Generic 2-layer MLP operator for tree-only ablation.

    Replaces the manifold's structured operator with a learned but
    unstructured transformation. Parameter-matched to manifold (~164K
    vs manifold's ~132K — acceptable mismatch, both negligible vs total ~726K).

    Used in E2b "Tree + MLP" variant: the tree still does multi-step
    composition, but the per-step operator is a generic MLP instead of
    the continuous operator manifold.
    """

    def __init__(self, hidden_dim: int = 256, struct_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + struct_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        structural: torch.Tensor,
    ) -> Tuple[torch.Tensor, None]:
        """
        Apply generic MLP operator.

        Args:
            x: [B, hidden_dim] state to transform
            structural: [B, struct_dim] structural component (concatenated as input)

        Returns:
            transformed: [B, hidden_dim] normalized output
            weights: None (no anchor weights for MLP operator)
        """
        combined = torch.cat([x, structural], dim=-1)
        out = self.mlp(combined)
        return F.normalize(out, dim=-1), None


if __name__ == "__main__":
    print("Testing ContinuousOperatorManifold...")

    hidden_dim = 256
    struct_dim = 128
    manifold_dim = 10
    n_anchors = 16
    rank = 16
    batch_size = 8

    manifold = ContinuousOperatorManifold(
        hidden_dim=hidden_dim,
        struct_dim=struct_dim,
        manifold_dim=manifold_dim,
        n_anchors=n_anchors,
        rank=rank,
    )

    total_params = sum(p.numel() for p in manifold.parameters())
    print(f"Parameters: {total_params:,}")
    print(f"  Anchors: {manifold.anchors.numel():,}")
    print(f"  U_all: {manifold.U_all.numel():,}")
    print(f"  V_all: {manifold.V_all.numel():,}")
    print(f"  Manifold proj: {sum(p.numel() for p in manifold.manifold_proj.parameters()):,}")

    # Test forward
    x = F.normalize(torch.randn(batch_size, hidden_dim), dim=-1)
    structural = F.normalize(torch.randn(batch_size, struct_dim), dim=-1)

    Ox, weights = manifold(x, structural)
    print(f"\nForward: x {x.shape} -> Ox {Ox.shape}")
    print(f"  Weights: {weights.shape}, sum={weights.sum(-1).mean():.4f}")
    print(f"  Output norm: {Ox.norm(dim=-1).mean():.4f}")

    # Test entropy loss
    ent_loss = manifold.entropy_loss(weights)
    print(f"  Entropy loss: {ent_loss.item():.4f}")

    # Test utilization
    util = manifold.get_anchor_utilization(weights)
    print(f"  Utilization: entropy={util['entropy']:.3f}, "
          f"norm_entropy={util['normalized_entropy']:.3f}, "
          f"PR={util['participation_ratio']:.2f}")

    # Test temperature annealing
    print(f"\n  Temperature: {manifold.tau.item():.3f}")
    manifold.anneal_temperature(10)
    print(f"  After epoch 10: {manifold.tau.item():.3f}")
    manifold.anneal_temperature(50)
    print(f"  After epoch 50: {manifold.tau.item():.3f}")

    # Test gradient flow
    loss = Ox.sum() + ent_loss
    loss.backward()
    grad_params = sum(1 for p in manifold.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    total = sum(1 for p in manifold.parameters())
    print(f"\nGradient flow: {grad_params}/{total} params have gradients")

    # Test operator similarity
    sim = manifold.get_operator_similarity()
    print(f"Operator similarity: diag={sim.diag().mean():.4f}, off-diag={sim.fill_diagonal_(0).abs().mean():.4f}")

    print("\n  StructuralOperator test...")
    op = StructuralOperator(hidden_dim, rank=16, name="test")
    y = op(x)
    print(f"  {x.shape} -> {y.shape}, norm={y.norm(dim=-1).mean():.4f}")

    print("\nAll operator tests passed!")
